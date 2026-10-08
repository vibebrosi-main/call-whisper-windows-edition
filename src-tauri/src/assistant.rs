//! Skąd biorą się podpowiedzi: most do lokalnego Claude Code (bez klucza,
//! w ramach subskrypcji) albo API zgodne z OpenAI. Port `ClaudeBridge.swift`
//! i `AssistantClient.swift`.

use crate::whisper_server::job;
#[cfg(windows)]
use crate::whisper_server::CREATE_NO_WINDOW;
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, ChildStdin, Command};
use tokio::sync::{mpsc, Mutex};

/// Ten sam tekst co na macOS — wspólny dla mostu i API.
pub const SYSTEM_PROMPT: &str = include_str!("prompts/system.txt");

/// Obrazek dołączony do pytania (zrzut ekranu), już przeskalowany.
#[derive(Clone)]
pub struct Image {
    pub base64: String,
    pub media_type: String,
}

// --- API ---

pub const API_BASE: &str = "https://api.experientiallabs.ai/v1";
pub const DEFAULT_API_MODEL: &str = "gemini-3.5-flash-lite";

/// Modele bramki z czasem do pierwszego tokenu zmierzonym na macOS.
pub const API_MODELS: &[(&str, &str)] = &[
    (
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash-lite (płatny, najszybszy)",
    ),
    ("mercury-2", "mercury-2 (płatny)"),
    (
        "claude-haiku-4.5",
        "claude-haiku-4.5 (płatny, dłuższe odpowiedzi)",
    ),
    ("gemini-3.1-flash-lite", "gemini-3.1-flash-lite (płatny)"),
    ("glm-4.7-flash", "glm-4.7-flash (płatny, rozwlekły)"),
    ("gemini-3.8-flash", "gemini-3.8-flash (płatny)"),
    ("openrouter-free", "openrouter-free (darmowy, wolny)"),
    (
        "minimax-m2.7-free",
        "minimax-m2.7-free (darmowy, bardzo wolny)",
    ),
];

/// Strumień odpowiedzi z `/chat/completions`. `on_delta(tekst, ttft_ms)`,
/// gdzie ttft jest tylko przy pierwszym kawałku.
pub async fn ask_api(
    prompt: &str,
    api_key: &str,
    model: &str,
    image: Option<&Image>,
    mut on_delta: impl FnMut(&str, Option<f64>),
) -> Result<(), String> {
    if api_key.is_empty() {
        return Err("Brak klucza API. Ustawienia → Podpowiedzi → Klucz API.".into());
    }
    let user: Value = match image {
        Some(img) => json!([
            { "type": "image_url", "image_url": { "url": format!("data:{};base64,{}", img.media_type, img.base64) } },
            { "type": "text", "text": prompt },
        ]),
        None => json!(prompt),
    };
    let body = json!({
        "model": if model.is_empty() { DEFAULT_API_MODEL } else { model },
        "stream": true,
        "max_tokens": 300,
        "messages": [
            { "role": "system", "content": SYSTEM_PROMPT },
            { "role": "user", "content": user },
        ],
    });
    let started = Instant::now();
    let response = reqwest::Client::builder()
        .timeout(Duration::from_secs(90))
        .build()
        .map_err(|e| e.to_string())?
        .post(format!("{API_BASE}/chat/completions"))
        .bearer_auth(api_key)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Nie udało się połączyć: {e}"))?;

    let status = response.status().as_u16();
    if status != 200 {
        let retry = response
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.parse::<f64>().ok());
        let text: String = response
            .text()
            .await
            .unwrap_or_default()
            .chars()
            .take(400)
            .collect();
        return Err(match status {
            401 | 403 => format!(
                "Bramka odrzuciła klucz (401). Sprawdź, czy jest wklejony w całości i bez spacji. {}",
                text.chars().take(160).collect::<String>()
            ),
            429 => format!(
                "Model darmowy jest chwilowo przeciążony (429).{}",
                retry.map(|r| format!(" Spróbuj za {} s.", r as i64)).unwrap_or_default()
            ),
            _ => format!("Model odpowiedział {status}: {}", text.chars().take(200).collect::<String>()),
        });
    }

    let mut first = true;
    let mut buffer = String::new();
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("Nie udało się połączyć: {e}"))?;
        buffer.push_str(&String::from_utf8_lossy(&chunk));
        while let Some(newline) = buffer.find('\n') {
            let line = buffer[..newline].trim().to_string();
            buffer.drain(..=newline);
            let Some(payload) = line.strip_prefix("data: ") else {
                continue;
            };
            if payload == "[DONE]" {
                return Ok(());
            }
            let Ok(json) = serde_json::from_str::<Value>(payload) else {
                continue;
            };
            let Some(content) = json["choices"][0]["delta"]["content"]
                .as_str()
                .filter(|c| !c.is_empty())
            else {
                continue;
            };
            let ttft = first.then(|| started.elapsed().as_secs_f64() * 1000.0);
            first = false;
            on_delta(content, ttft);
        }
    }
    Ok(())
}

// --- most do Claude Code ---

/// Narzędzia wyłączone: sufler ma odpowiadać, a nie czytać pliki i odpalać
/// polecenia w trakcie rozmowy.
const DISABLED_TOOLS: &str =
    "Bash,Read,Write,Edit,Glob,Grep,WebFetch,WebSearch,Task,TodoWrite,NotebookEdit";
/// Po tylu pytaniach sesja idzie do wymiany: kontekst rośnie z każdym
/// pytaniem i odpowiedzi zwalniają.
const MAX_USES: u32 = 3;

/// Jak uruchomić `claude`: plik wykonywalny albo `node cli.js` (instalacja
/// z npm daje `claude.cmd`, a przez cmd.exe nie da się przekazać promptu
/// systemowego z nowymi liniami).
#[derive(Clone)]
pub struct Cli {
    program: PathBuf,
    prefix: Vec<String>,
}

pub fn locate_cli() -> Option<Cli> {
    let home = dirs::home_dir().unwrap_or_default();
    let exe = if cfg!(windows) {
        "claude.exe"
    } else {
        "claude"
    };
    let mut candidates = vec![
        home.join(".local").join("bin").join(exe),
        home.join(".claude").join("local").join(exe),
    ];
    if let Some(path) = std::env::var_os("PATH") {
        candidates.extend(std::env::split_paths(&path).map(|d| d.join(exe)));
    }
    if let Some(found) = candidates.into_iter().find(|p| p.is_file()) {
        return Some(Cli {
            program: found,
            prefix: Vec::new(),
        });
    }
    // npm: %APPDATA%\npm\node_modules\@anthropic-ai\claude-code\cli.js
    let npm = dirs::data_dir()?
        .join("npm")
        .join("node_modules")
        .join("@anthropic-ai")
        .join("claude-code")
        .join("cli.js");
    if npm.is_file() {
        let node = std::env::var_os("PATH").and_then(|path| {
            std::env::split_paths(&path)
                .map(|d| d.join(if cfg!(windows) { "node.exe" } else { "node" }))
                .find(|p| p.is_file())
        })?;
        return Some(Cli {
            program: node,
            prefix: vec![npm.to_string_lossy().into_owned()],
        });
    }
    None
}

struct Session {
    child: Child,
    stdin: ChildStdin,
    lines: mpsc::UnboundedReceiver<Value>,
    uses: u32,
}

impl Session {
    fn spawn(cli: &Cli, model: &str) -> Result<Self, String> {
        let mut command = Command::new(&cli.program);
        command
            .args(&cli.prefix)
            .args([
                "-p",
                "--input-format",
                "stream-json",
                "--output-format",
                "stream-json",
            ])
            .args([
                "--include-partial-messages",
                "--verbose",
                "--no-session-persistence",
            ])
            .args([
                "--permission-mode",
                "dontAsk",
                "--disallowed-tools",
                DISABLED_TOOLS,
            ])
            .args(["--append-system-prompt", SYSTEM_PROMPT]);
        if !model.is_empty() {
            command.args(["--model", model]);
        }
        command
            .current_dir(std::env::temp_dir())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        #[cfg(windows)]
        command.creation_flags(CREATE_NO_WINDOW);
        let mut child = command
            .spawn()
            .map_err(|e| format!("Nie udało się uruchomić claude: {e}"))?;
        if let Some(id) = child.id() {
            job::assign_pid(id);
        }
        let stdin = child.stdin.take().ok_or("brak stdin")?;
        let stdout = child.stdout.take().ok_or("brak stdout")?;
        let (tx, lines) = mpsc::unbounded_channel();
        tokio::spawn(async move {
            let mut reader = BufReader::new(stdout).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                if let Ok(value) = serde_json::from_str::<Value>(line.trim()) {
                    if tx.send(value).is_err() {
                        break;
                    }
                }
            }
        });
        Ok(Self {
            child,
            stdin,
            lines,
            uses: 0,
        })
    }

    async fn run(
        &mut self,
        prompt: &str,
        image: Option<&Image>,
        timeout: Duration,
        mut on_delta: impl FnMut(&str),
    ) -> Result<String, String> {
        let mut content = Vec::new();
        if let Some(img) = image {
            content.push(json!({ "type": "image", "source": { "type": "base64", "media_type": img.media_type, "data": img.base64 } }));
        }
        content.push(json!({ "type": "text", "text": prompt }));
        let mut line = serde_json::to_vec(
            &json!({ "type": "user", "message": { "role": "user", "content": content } }),
        )
        .map_err(|e| e.to_string())?;
        line.push(b'\n');
        self.stdin
            .write_all(&line)
            .await
            .map_err(|e| e.to_string())?;
        self.stdin.flush().await.map_err(|e| e.to_string())?;

        let mut text = String::new();
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            let message = match tokio::time::timeout_at(deadline, self.lines.recv()).await {
                Err(_) => {
                    return Err(format!(
                        "Claude nie odpowiedział w {} s.",
                        timeout.as_secs()
                    ))
                }
                Ok(None) => {
                    let code = self
                        .child
                        .try_wait()
                        .ok()
                        .flatten()
                        .and_then(|s| s.code())
                        .unwrap_or(-1);
                    return Err(format!("Proces claude zakończył się (kod {code})."));
                }
                Ok(Some(message)) => message,
            };
            let delta = message["event"]["delta"]["text"]
                .as_str()
                .or(message["delta"]["text"].as_str());
            if let Some(delta) = delta.filter(|d| !d.is_empty()) {
                text.push_str(delta);
                on_delta(delta);
                continue;
            }
            if message["type"] == "result" {
                if text.is_empty() {
                    text = message["result"].as_str().unwrap_or("").to_string();
                }
                return if message["is_error"] == true {
                    Err(if text.is_empty() {
                        "Claude zwrócił błąd.".into()
                    } else {
                        text
                    })
                } else {
                    Ok(text.trim().to_string())
                };
            }
        }
    }
}

/// Utrzymuje rozgrzaną sesję zapasową: zimny start CLI to ~3,9 s do
/// pierwszego tokenu, ciepły ~1,8 s. Ten koszt płacimy w tle.
pub struct ClaudeBridge {
    state: Mutex<BridgeState>,
}

#[derive(Default)]
struct BridgeState {
    active: Option<Session>,
    spare: Option<Session>,
    warming: bool,
    stopped: bool,
    model: String,
    cli: Option<Cli>,
}

impl ClaudeBridge {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            state: Mutex::new(BridgeState {
                stopped: true,
                ..Default::default()
            }),
        })
    }

    pub async fn start(self: &Arc<Self>, model: &str) -> Result<(), String> {
        let cli = locate_cli().ok_or(
            "Nie znalazłem CLI `claude`. Zainstaluj Claude Code albo wybierz model przez API.",
        )?;
        let mut state = self.state.lock().await;
        if state.model != model {
            state.active = None;
            state.spare = None;
        }
        state.model = model.to_string();
        state.cli = Some(cli);
        state.stopped = false;
        Ok(())
    }

    pub async fn stop(&self) {
        let mut state = self.state.lock().await;
        state.stopped = true;
        state.active = None;
        state.spare = None;
    }

    /// Rozgrzewa sesję zapasową w tle.
    pub fn warmup(self: &Arc<Self>) {
        let this = self.clone();
        tokio::spawn(async move { this.replenish().await });
    }

    async fn replenish(self: &Arc<Self>) {
        let (cli, model) = {
            let mut state = self.state.lock().await;
            if state.stopped || state.spare.is_some() || state.warming {
                return;
            }
            let Some(cli) = state.cli.clone() else { return };
            state.warming = true;
            (cli, state.model.clone())
        };
        let session = match Session::spawn(&cli, &model) {
            Ok(mut session) => session
                .run(
                    "Odpowiedz dokładnie jednym słowem: gotowy",
                    None,
                    Duration::from_secs(90),
                    |_| {},
                )
                .await
                .ok()
                .map(|_| session),
            Err(_) => None,
        };
        let mut state = self.state.lock().await;
        state.warming = false;
        if !state.stopped {
            state.spare = session;
        }
    }

    pub async fn ask(
        self: &Arc<Self>,
        prompt: &str,
        image: Option<&Image>,
        timeout: Duration,
        on_delta: impl FnMut(&str),
    ) -> Result<String, String> {
        let mut session = {
            let mut state = self.state.lock().await;
            if state.stopped {
                return Err("Most do Claude Code jest wyłączony.".into());
            }
            match state.active.take().or_else(|| state.spare.take()) {
                Some(session) => session,
                None => {
                    let cli = state.cli.clone().ok_or("Nie znalazłem CLI `claude`.")?;
                    Session::spawn(&cli, &state.model)?
                }
            }
        };
        let result = session.run(prompt, image, timeout, on_delta).await;
        session.uses += 1;
        {
            let mut state = self.state.lock().await;
            if result.is_ok() && session.uses < MAX_USES && !state.stopped && state.active.is_none()
            {
                state.active = Some(session);
            }
        }
        self.warmup();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Prawdziwe zapytanie przez lokalne CLI: `cargo test -p call-whisper -- --ignored bridge`.
    #[tokio::test]
    #[ignore]
    async fn bridge_answers_and_reuses_session() {
        let bridge = ClaudeBridge::new();
        bridge.start("haiku").await.expect("claude CLI");
        let mut streamed = String::new();
        let answer = bridge
            .ask(
                "Ile to 2+2? Odpowiedz samą liczbą.",
                None,
                Duration::from_secs(90),
                |d| streamed.push_str(d),
            )
            .await
            .expect("odpowiedź");
        assert!(answer.contains('4'), "{answer}");
        assert_eq!(streamed.trim(), answer);
        let second = bridge
            .ask("A 3+3? Samą liczbą.", None, Duration::from_secs(90), |_| {})
            .await
            .expect("druga");
        assert!(second.contains('6'), "{second}");
        bridge.stop().await;
    }
}
