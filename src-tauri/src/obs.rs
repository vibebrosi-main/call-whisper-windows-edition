//! Połączenie z OBS przez obs-websocket v5 — port `OBSLink.swift`.
//!
//! „Słuchaj" włącza nagrywanie w OBS, „Zatrzymaj" je kończy, a transkrypt
//! ląduje obok filmu. Działa też odwrotnie: nagranie włączone w OBS włącza
//! nasłuch, z czasem liczonym od startu nagrania.

use cw_core::obs::{Obs, RecordEvent, ServerConfig};
use cw_core::time::now_ms;
use futures_util::{SinkExt, StreamExt};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{mpsc, oneshot};
use tokio_tungstenite::tungstenite::Message;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    Off,
    Waiting,
    ServerDisabled,
    WrongPassword,
    Connected,
    Recording,
}

/// Zdarzenia nagrania, którego nie zamówiliśmy (włączone ręcznie w OBS).
pub enum Event {
    RecordStarted(f64),
    RecordStopped(Option<String>),
}

enum Command {
    Start(oneshot::Sender<Option<f64>>),
    Stop(oneshot::Sender<Option<String>>),
}

pub struct ObsLink {
    state: Arc<Mutex<State>>,
    commands: Mutex<Option<mpsc::UnboundedSender<Command>>>,
    events: mpsc::UnboundedSender<Event>,
    task: Mutex<Option<tokio::task::JoinHandle<()>>>,
}

const STATUS_REQUEST: &str = "cw-record-status";

impl ObsLink {
    pub fn new(events: mpsc::UnboundedSender<Event>) -> Arc<Self> {
        Arc::new(Self {
            state: Arc::new(Mutex::new(State::Off)),
            commands: Mutex::new(None),
            events,
            task: Mutex::new(None),
        })
    }

    pub fn state(&self) -> State {
        *self.state.lock().unwrap()
    }

    pub fn start(&self) {
        if self.task.lock().unwrap().is_some() {
            return;
        }
        let (tx, rx) = mpsc::unbounded_channel();
        *self.commands.lock().unwrap() = Some(tx);
        *self.state.lock().unwrap() = State::Waiting;
        let state = self.state.clone();
        let events = self.events.clone();
        *self.task.lock().unwrap() = Some(tokio::spawn(run(state, events, rx)));
    }

    pub fn stop(&self) {
        if let Some(task) = self.task.lock().unwrap().take() {
            task.abort();
        }
        self.commands.lock().unwrap().take();
        *self.state.lock().unwrap() = State::Off;
    }

    /// Włącza nagrywanie i zwraca jego początek (ms epoki) — od niego liczą
    /// się czasy w transkrypcie, żeby 00:01:05 znaczyło 00:01:05 w filmie.
    /// OBS niewłączony uruchamiamy sami.
    pub async fn start_recording(&self) -> Result<f64, String> {
        self.start();
        self.ensure_connected().await?;
        let (tx, rx) = oneshot::channel();
        self.send(Command::Start(tx))?;
        match tokio::time::timeout(Duration::from_secs(10), rx).await {
            Ok(Ok(Some(origin))) => Ok(origin),
            _ => Err("OBS nie odpowiedział na czas. Nasłuch działa bez nagrywania wideo.".into()),
        }
    }

    /// Kończy nagrywanie i zwraca ścieżkę pliku wideo.
    pub async fn stop_recording(&self) -> Option<String> {
        if self.state() != State::Recording {
            return None;
        }
        let (tx, rx) = oneshot::channel();
        self.send(Command::Stop(tx)).ok()?;
        tokio::time::timeout(Duration::from_secs(15), rx)
            .await
            .ok()?
            .ok()?
    }

    fn send(&self, command: Command) -> Result<(), String> {
        self.commands
            .lock()
            .unwrap()
            .as_ref()
            .ok_or("OBS: połączenie wyłączone")?
            .send(command)
            .map_err(|_| "OBS: połączenie wyłączone".to_string())
    }

    async fn ensure_connected(&self) -> Result<(), String> {
        match self.state() {
            State::Connected | State::Recording => return Ok(()),
            State::WrongPassword => return Err(WRONG_PASSWORD.into()),
            _ => {}
        }
        if !process::obs_running() {
            let exe = process::obs_executable()
                .ok_or("Nie znaleziono OBS. Nasłuch działa bez nagrywania wideo.")?;
            // OBS jeszcze nie działa, więc możemy bezpiecznie włączyć serwer
            // w jego konfiguracji — przeczyta ją przy starcie.
            if let Some(path) = ServerConfig::default_path() {
                ServerConfig::enable_server(&path);
            }
            process::launch(&exe).map_err(|e| format!("Nie udało się uruchomić OBS: {e}"))?;
        } else if ServerConfig::default_path()
            .and_then(|p| ServerConfig::load(&p))
            .is_some_and(|c| !c.enabled)
        {
            return Err(SERVER_DISABLED.into());
        }
        // OBS startuje kilka sekund; serwer wstaje razem z nim.
        for _ in 0..60 {
            match self.state() {
                State::Connected | State::Recording => return Ok(()),
                State::WrongPassword => return Err(WRONG_PASSWORD.into()),
                _ => tokio::time::sleep(Duration::from_millis(500)).await,
            }
        }
        Err("OBS nie odpowiedział na czas. Nasłuch działa bez nagrywania wideo.".into())
    }
}

const WRONG_PASSWORD: &str =
    "OBS odrzucił hasło WebSocket. Zapisz ustawienia serwera w OBS jeszcze raz.";
const SERVER_DISABLED: &str =
    "W OBS włącz serwer: Narzędzia → Ustawienia serwera WebSocket → Włącz serwer WebSocket.";

/// Pętla połączenia: łączy, a po zerwaniu ponawia co 3 s.
async fn run(
    state: Arc<Mutex<State>>,
    events: mpsc::UnboundedSender<Event>,
    mut commands: mpsc::UnboundedReceiver<Command>,
) {
    let set = |s: State| *state.lock().unwrap() = s;
    loop {
        let config = ServerConfig::default_path().and_then(|p| ServerConfig::load(&p));
        match config {
            None => set(State::Waiting),
            Some(c) if !c.enabled => set(if process::obs_running() {
                State::ServerDisabled
            } else {
                State::Waiting
            }),
            Some(c) => {
                if let Ok((socket, _)) =
                    tokio_tungstenite::connect_async(format!("ws://127.0.0.1:{}", c.port)).await
                {
                    let was_recording = session(
                        socket,
                        c.password.unwrap_or_default(),
                        &state,
                        &events,
                        &mut commands,
                    )
                    .await;
                    // OBS zamknięty w trakcie nagrywania też kończy nagranie.
                    if was_recording {
                        let _ = events.send(Event::RecordStopped(None));
                    }
                }
            }
        }
        tokio::time::sleep(Duration::from_secs(3)).await;
    }
}

/// Jedno połączenie. Zwraca, czy w chwili zerwania trwało nagrywanie.
async fn session(
    socket: tokio_tungstenite::WebSocketStream<
        tokio_tungstenite::MaybeTlsStream<tokio::net::TcpStream>,
    >,
    password: String,
    state: &Arc<Mutex<State>>,
    events: &mpsc::UnboundedSender<Event>,
    commands: &mut mpsc::UnboundedReceiver<Command>,
) -> bool {
    let (mut write, mut read) = socket.split();
    let set = |s: State| *state.lock().unwrap() = s;
    let get = || *state.lock().unwrap();
    let mut start_waiter: Option<oneshot::Sender<Option<f64>>> = None;
    let mut stop_waiter: Option<oneshot::Sender<Option<String>>> = None;

    loop {
        tokio::select! {
            message = read.next() => {
                let text = match message {
                    Some(Ok(Message::Text(text))) => text.to_string(),
                    Some(Ok(Message::Close(frame))) => {
                        // Kod 4009 to złe hasło; reszta = OBS jeszcze nie wstał.
                        let code: u16 = frame.map(|f| f.code.into()).unwrap_or(0);
                        let was = get() == State::Recording;
                        set(if code == 4009 { State::WrongPassword } else { State::Waiting });
                        return was;
                    }
                    Some(Ok(_)) => continue,
                    _ => {
                        let was = get() == State::Recording;
                        if get() != State::WrongPassword { set(State::Waiting); }
                        return was;
                    }
                };
                let Ok(json) = serde_json::from_str::<Value>(&text) else { continue };
                let d = &json["d"];
                match json["op"].as_i64() {
                    Some(0) => {
                        let mut identify = json!({ "rpcVersion": 1, "eventSubscriptions": Obs::OUTPUTS_EVENT_SUBSCRIPTION });
                        if let (Some(salt), Some(challenge)) = (d["authentication"]["salt"].as_str(), d["authentication"]["challenge"].as_str()) {
                            identify["authentication"] = json!(Obs::authentication(&password, salt, challenge));
                        }
                        let _ = write.send(Message::Text(json!({ "op": 1, "d": identify }).to_string().into())).await;
                    }
                    Some(2) => {
                        set(State::Connected);
                        // Nagrywanie mogło trwać, zanim się połączyliśmy.
                        let request = json!({ "op": 6, "d": { "requestType": "GetRecordStatus", "requestId": STATUS_REQUEST } });
                        let _ = write.send(Message::Text(request.to_string().into())).await;
                    }
                    Some(7) => {
                        if d["requestId"] == STATUS_REQUEST && d["responseData"]["outputActive"] == true {
                            let elapsed = d["responseData"]["outputDuration"].as_f64().unwrap_or(0.0);
                            started(now_ms() - elapsed, state, events, &mut start_waiter);
                        }
                    }
                    _ => match Obs::record_event(&json) {
                        Some(RecordEvent::Started { .. }) => started(now_ms(), state, events, &mut start_waiter),
                        Some(RecordEvent::Stopped { path }) => {
                            if get() == State::Recording {
                                set(State::Connected);
                                match stop_waiter.take() {
                                    Some(waiter) => { let _ = waiter.send(path); }
                                    None => { let _ = events.send(Event::RecordStopped(path)); }
                                }
                            }
                        }
                        None => {}
                    },
                }
            }
            command = commands.recv() => {
                let Some(command) = command else { return false };
                let (request_type, id) = match command {
                    Command::Start(tx) => {
                        if get() == State::Recording { let _ = tx.send(None); continue; }
                        start_waiter = Some(tx);
                        ("StartRecord", "cw-start")
                    }
                    Command::Stop(tx) => { stop_waiter = Some(tx); ("StopRecord", "cw-stop") }
                };
                let request = json!({ "op": 6, "d": { "requestType": request_type, "requestId": id } });
                let _ = write.send(Message::Text(request.to_string().into())).await;
            }
        }
    }
}

fn started(
    origin: f64,
    state: &Arc<Mutex<State>>,
    events: &mpsc::UnboundedSender<Event>,
    waiter: &mut Option<oneshot::Sender<Option<f64>>>,
) {
    let mut s = state.lock().unwrap();
    if *s == State::Recording {
        return;
    }
    *s = State::Recording;
    match waiter.take() {
        // Nagranie, o które sami poprosiliśmy.
        Some(waiter) => {
            let _ = waiter.send(Some(origin));
        }
        None => {
            let _ = events.send(Event::RecordStarted(origin));
        }
    }
}

/// Wykrywanie i uruchamianie OBS na Windows.
mod process {
    use std::path::PathBuf;

    pub fn obs_executable() -> Option<PathBuf> {
        let mut roots = Vec::new();
        for var in ["ProgramFiles", "ProgramFiles(x86)", "ProgramW6432"] {
            if let Some(dir) = std::env::var_os(var) {
                roots.push(PathBuf::from(dir));
            }
        }
        roots
            .into_iter()
            .map(|r| {
                r.join("obs-studio")
                    .join("bin")
                    .join("64bit")
                    .join("obs64.exe")
            })
            .find(|p| p.is_file())
    }

    /// OBS szuka swoich plików względem katalogu roboczego — musi nim być
    /// katalog z obs64.exe, inaczej nie wstaje.
    pub fn launch(exe: &std::path::Path) -> std::io::Result<()> {
        std::process::Command::new(exe)
            .current_dir(exe.parent().unwrap_or(std::path::Path::new(".")))
            .args(["--minimize-to-tray", "--disable-shutdown-check"])
            .spawn()
            .map(|_| ())
    }

    #[cfg(windows)]
    pub fn obs_running() -> bool {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Diagnostics::ToolHelp::*;
        unsafe {
            let snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
            if snapshot == INVALID_HANDLE_VALUE {
                return false;
            }
            let mut entry: PROCESSENTRY32W = std::mem::zeroed();
            entry.dwSize = std::mem::size_of::<PROCESSENTRY32W>() as u32;
            let mut found = false;
            if Process32FirstW(snapshot, &mut entry) != 0 {
                loop {
                    let len = entry
                        .szExeFile
                        .iter()
                        .position(|&c| c == 0)
                        .unwrap_or(entry.szExeFile.len());
                    let name = String::from_utf16_lossy(&entry.szExeFile[..len]);
                    if name.eq_ignore_ascii_case("obs64.exe") {
                        found = true;
                        break;
                    }
                    if Process32NextW(snapshot, &mut entry) == 0 {
                        break;
                    }
                }
            }
            CloseHandle(snapshot);
            found
        }
    }

    #[cfg(not(windows))]
    pub fn obs_running() -> bool {
        false
    }
}
