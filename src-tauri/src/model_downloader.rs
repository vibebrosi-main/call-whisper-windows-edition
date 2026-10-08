//! Pobieranie modeli whisper.cpp do %LOCALAPPDATA%\call-whisper\models.
//!
//! Instalator nie wiezie modelu (0,5-1,6 GB) — pobieramy go przy pierwszym
//! nasłuchu, z paskiem postępu.

use crate::settings;
use crate::whisper_server::model_path;
use futures_util::StreamExt;
use serde::Serialize;
use std::io::Write;
use std::time::{Duration, Instant};

const BASE_URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main";

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct KnownModel {
    pub id: &'static str,
    pub bytes: u64,
    pub note: &'static str,
}

/// Rozmiary z repozytorium — do pokazania, na co się użytkownik pisze.
pub const KNOWN: &[KnownModel] = &[
    KnownModel {
        id: "small",
        bytes: 487_601_967,
        note: "domyślny, 13,3 % błędnych słów po polsku, ~930 ms",
    },
    KnownModel {
        id: "large-v3-turbo",
        bytes: 1_624_555_275,
        note: "dokładniejszy: 10,0 %, ale ~1,6 s i 1,6 GB pamięci",
    },
    KnownModel {
        id: "base",
        bytes: 147_951_465,
        note: "najmniejszy, wyraźnie gorszy po polsku",
    },
];

#[derive(Clone, Copy, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Progress {
    pub fraction: f64,
    pub received_bytes: u64,
    pub total_bytes: u64,
}

pub fn is_installed(model: &str) -> bool {
    model_path(model).is_file()
}

/// Pobiera model, jeśli go nie ma. Zapis do pliku `.part` i zmiana nazwy na
/// końcu, więc przerwane pobieranie nie zostawia „zainstalowanego" śmiecia.
pub async fn download(model: &str, on_progress: impl Fn(Progress)) -> anyhow::Result<()> {
    let known = KNOWN
        .iter()
        .find(|m| m.id == model)
        .ok_or_else(|| anyhow::anyhow!("Nie znam modelu {model}."))?;
    let target = model_path(model);
    if target.is_file() {
        return Ok(());
    }
    std::fs::create_dir_all(settings::models_dir())?;

    let response = reqwest::Client::new()
        .get(format!("{BASE_URL}/ggml-{model}.bin"))
        .send()
        .await?;
    if !response.status().is_success() {
        anyhow::bail!("Pobieranie zwróciło {}.", response.status().as_u16());
    }
    let total = response.content_length().unwrap_or(known.bytes);
    let partial = target.with_extension("bin.part");
    let mut file = std::fs::File::create(&partial)?;
    let mut received: u64 = 0;
    let mut last_report = Instant::now() - Duration::from_secs(1);
    let mut stream = response.bytes_stream();
    while let Some(chunk) = stream.next().await {
        let chunk = chunk?;
        file.write_all(&chunk)?;
        received += chunk.len() as u64;
        // Co 200 ms wystarczy — pasek postępu nie potrzebuje więcej.
        if last_report.elapsed() >= Duration::from_millis(200) {
            last_report = Instant::now();
            on_progress(Progress {
                fraction: received as f64 / total as f64,
                received_bytes: received,
                total_bytes: total,
            });
        }
    }
    file.flush()?;
    drop(file);
    if received < 1_000_000 || received < total {
        let _ = std::fs::remove_file(&partial);
        anyhow::bail!("Pobieranie przerwane — plik jest niekompletny.");
    }
    std::fs::rename(&partial, &target)?;
    on_progress(Progress {
        fraction: 1.0,
        received_bytes: received,
        total_bytes: total,
    });
    Ok(())
}
