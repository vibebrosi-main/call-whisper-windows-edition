//! Uruchamia i pilnuje lokalnego `whisper-server.exe` (whisper.cpp).
//!
//! Instalator wiezie dwie wersje: Vulkan (każde GPU: Nvidia, AMD, Intel)
//! i CPU. Najpierw Vulkan; gdy nie wstanie (stary sterownik, maszyna
//! wirtualna bez GPU), zapasowo CPU.

use crate::settings;
use crate::whisper_client::WhisperClient;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct Config {
    pub model: String,
    pub port: u16,
    pub language: String,
    pub threads: usize,
}

pub fn model_path(model: &str) -> PathBuf {
    settings::models_dir().join(format!("ggml-{model}.bin"))
}

/// Warianty silnika w kolejności prób.
fn variants(vendor: &Path) -> Vec<(&'static str, PathBuf)> {
    ["vulkan", "cpu"]
        .into_iter()
        .map(|v| (v, vendor.join("whisper").join(v).join(exe("whisper-server"))))
        .filter(|(_, p)| p.exists())
        .collect()
}

fn exe(name: &str) -> String {
    if cfg!(windows) { format!("{name}.exe") } else { name.to_string() }
}

pub struct WhisperServer {
    vendor: PathBuf,
    /// Procesy per port.
    processes: HashMap<u16, Child>,
    /// Wariant, który ostatnio wstał — kolejny start zaczyna od niego.
    pub variant: Option<&'static str>,
}

impl WhisperServer {
    pub fn new(vendor: PathBuf) -> Self {
        Self { vendor, processes: HashMap::new(), variant: None }
    }

    pub fn binary_available(&self) -> bool {
        !variants(&self.vendor).is_empty()
    }

    /// Startuje serwer, jeśli jeszcze nie odpowiada. Czeka, aż zacznie
    /// odpowiadać. `Ok(false)` = już działał (np. uruchomiony ręcznie).
    pub async fn ensure_running(&mut self, config: &Config) -> anyhow::Result<bool> {
        let client = WhisperClient::new(config.port, &config.language, 15);
        if client.health().await {
            return Ok(false);
        }
        let model = model_path(&config.model);
        if !model.is_file() {
            anyhow::bail!(
                "Brak modelu ggml-{}.bin w {}.",
                config.model,
                settings::models_dir().display()
            );
        }
        let mut candidates = variants(&self.vendor);
        if candidates.is_empty() {
            anyhow::bail!("Brak silnika mowy (whisper-server.exe) w instalacji. Zainstaluj aplikację ponownie.");
        }
        if let Some(last) = self.variant {
            candidates.sort_by_key(|(v, _)| *v != last);
        }

        let mut last_error = String::new();
        for (variant, binary) in candidates {
            match self.spawn(&binary, config, &model, &client).await {
                Ok(()) => {
                    self.variant = Some(variant);
                    return Ok(true);
                }
                Err(err) => last_error = format!("{variant}: {err}"),
            }
        }
        anyhow::bail!("whisper-server nie wystartował: {last_error}")
    }

    async fn spawn(&mut self, binary: &Path, config: &Config, model: &Path, client: &WhisperClient) -> anyhow::Result<()> {
        // Wyjście do pliku, nie do potoku: potok, którego nikt nie czyta,
        // zapycha się po 64 kB i serwer staje w połowie rozmowy.
        let log_dir = settings::log_dir();
        std::fs::create_dir_all(&log_dir)?;
        let log_path = log_dir.join(format!("whisper-{}.log", config.port));
        let log = std::fs::File::create(&log_path)?;

        let mut command = Command::new(binary);
        command
            .args(["--model", &model.to_string_lossy()])
            .args(["--port", &config.port.to_string()])
            .args(["--host", "127.0.0.1"])
            .args(["--language", &config.language])
            .args(["--threads", &config.threads.to_string()])
            // Segmenty cięte na granicy słowa, nie tokenu.
            .arg("--split-on-word")
            .current_dir(binary.parent().unwrap_or(Path::new(".")))
            .stdin(Stdio::null())
            .stdout(Stdio::from(log.try_clone()?))
            .stderr(Stdio::from(log));
        hide_console(&mut command);

        let mut child = command.spawn()?;
        job::assign(&child);

        // Model `small` wczytuje się w ~1 s, ale Vulkan kompiluje shadery przy
        // pierwszym starcie, a `large-v3-turbo` ładuje się dłużej. 45 s zapasu.
        for _ in 0..180 {
            if client.health().await {
                self.processes.insert(config.port, child);
                return Ok(());
            }
            if let Ok(Some(status)) = child.try_wait() {
                let output = std::fs::read_to_string(&log_path).unwrap_or_default();
                let tail: String = output.chars().rev().take(200).collect::<Vec<_>>().into_iter().rev().collect();
                anyhow::bail!("zakończył się ({status}): {tail}");
            }
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
        let _ = child.kill();
        anyhow::bail!("brak odpowiedzi po 45 s")
    }

    /// Zatrzymujemy tylko procesy, które sami uruchomiliśmy.
    pub fn stop(&mut self) {
        for (_, mut child) in self.processes.drain() {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
}

impl Drop for WhisperServer {
    fn drop(&mut self) {
        self.stop();
    }
}

#[cfg(windows)]
pub fn hide_console(command: &mut Command) {
    use std::os::windows::process::CommandExt;
    const CREATE_NO_WINDOW: u32 = 0x0800_0000;
    command.creation_flags(CREATE_NO_WINDOW);
}

#[cfg(not(windows))]
pub fn hide_console(_command: &mut Command) {}

/// Job Object z KILL_ON_JOB_CLOSE: procesy potomne giną razem z aplikacją,
/// także przy awarii albo zabiciu z Menedżera zadań. Na macOS trzeba było
/// do tego szukać sierot po `ps`; tu robi to system.
#[cfg(windows)]
pub mod job {
    use std::os::windows::io::AsRawHandle;
    use std::sync::OnceLock;
    use windows_sys::Win32::System::JobObjects::*;

    struct Job(isize);
    unsafe impl Send for Job {}
    unsafe impl Sync for Job {}

    fn handle() -> Option<isize> {
        static JOB: OnceLock<Option<Job>> = OnceLock::new();
        JOB.get_or_init(|| unsafe {
            let job = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if job.is_null() {
                return None;
            }
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            SetInformationJobObject(
                job,
                JobObjectExtendedLimitInformation,
                &info as *const _ as *const _,
                std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            );
            Some(Job(job as isize))
        })
        .as_ref()
        .map(|j| j.0)
    }

    pub fn assign(child: &std::process::Child) {
        if let Some(job) = handle() {
            unsafe {
                AssignProcessToJobObject(job as _, child.as_raw_handle() as _);
            }
        }
    }
}

#[cfg(not(windows))]
pub mod job {
    pub fn assign(_child: &std::process::Child) {}
}
