//! Import nagrania albo wideo do transkryptu — port `MediaImport.swift`.
//! Na macOS dekodował AVFoundation; tu ffmpeg z instalatora, który od razu
//! zmiksowuje wszystkie ścieżki do 16 kHz mono.

use crate::audio::ASR_SAMPLE_RATE;
use crate::whisper_client::{Quality, WhisperClient};
use crate::whisper_server::hide_console;
use cw_core::markdown::SessionMeta;
use cw_core::speaker_turns::{SpeakerTurns, TimedText};
use cw_core::transcript_store::Segment;
use std::ops::Range;
use std::path::Path;
use std::process::{Command, Stdio};

pub struct ImportResult {
    pub segments: Vec<Segment>,
    pub meta: SessionMeta,
    /// Dlaczego rozpoznawanie głosów się nie odbyło, choć było zamówione.
    pub diarization_note: Option<String>,
}

/// Dekoduje dźwięk do 16 kHz mono f32. Mix, a nie pojedyncza ścieżka:
/// nagrania rozmów potrafią mieć osobne ścieżki na każdą stronę.
pub fn decode(ffmpeg: &Path, file: &Path) -> anyhow::Result<Vec<f32>> {
    let mut command = Command::new(ffmpeg);
    command
        .args(["-nostdin", "-hide_banner", "-loglevel", "error", "-i"])
        .arg(file)
        .args(["-vn", "-ac", "1", "-ar", "16000", "-f", "f32le", "-"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    hide_console(&mut command);
    let output = command
        .output()
        .map_err(|e| anyhow::anyhow!("Nie udało się uruchomić ffmpeg: {e}"))?;
    if !output.status.success() {
        let err = String::from_utf8_lossy(&output.stderr);
        anyhow::bail!(
            "Nie udało się odczytać dźwięku: {}",
            err.trim().chars().take(200).collect::<String>()
        );
    }
    let samples: Vec<f32> = output
        .stdout
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    if samples.is_empty() {
        let name = file
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        anyhow::bail!("{name} nie ma ścieżki dźwiękowej.");
    }
    Ok(samples)
}

/// Kawałki do transkrypcji, cięte w najcichszym miejscu: najcichsze 100 ms
/// w ostatnich `window` próbkach kawałka. Cięcie co równe 10 minut trafiało
/// w połowę słowa, które ginęło albo przekręcało się po obu stronach.
pub fn chunk_bounds(
    count: usize,
    target: usize,
    window: usize,
    samples: &[f32],
) -> Vec<Range<usize>> {
    let mut bounds = Vec::new();
    let mut start = 0;
    let hop = (ASR_SAMPLE_RATE / 10.0) as usize;
    while start < count {
        let mut end = count.min(start + target);
        if end < count {
            let (mut best, mut best_energy) = (end, f32::MAX);
            let mut probe = (start + hop).max(end.saturating_sub(window));
            while probe + hop <= end {
                let energy: f32 = samples[probe..probe + hop].iter().map(|s| s * s).sum();
                if energy < best_energy {
                    best_energy = energy;
                    best = probe + hop / 2;
                }
                probe += hop;
            }
            end = best;
        }
        bounds.push(start..end);
        start = end;
    }
    bounds
}

/// Cały import: dekodowanie -> whisper -> segmenty. `whisper-server` ma już
/// działać — uruchamia go wołający.
pub async fn run(
    ffmpeg: &Path,
    file: &Path,
    whisper_port: u16,
    language: &str,
    diarizer: Option<crate::diarization::Engine>,
    progress: impl Fn(String),
) -> anyhow::Result<ImportResult> {
    let name = file
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    progress(format!("Odczytuję dźwięk z {name}…"));
    let (ffmpeg_path, file_path) = (ffmpeg.to_path_buf(), file.to_path_buf());
    let pcm = tokio::task::spawn_blocking(move || decode(&ffmpeg_path, &file_path)).await??;
    let seconds = pcm.len() as f64 / ASR_SAMPLE_RATE;

    // Dwie minuty na kawałek: kilka sekund inferencji na `small`, więc pasek
    // postępu żyje i nie zbliżamy się do limitu czasu.
    let rate = ASR_SAMPLE_RATE as usize;
    let bounds = chunk_bounds(pcm.len(), 120 * rate, 10 * rate, &pcm);
    let client = WhisperClient::new(whisper_port, language, 600);
    let mut texts: Vec<TimedText> = Vec::new();
    let mut context = String::new();
    for (index, range) in bounds.iter().enumerate() {
        progress(format!(
            "Transkrybuję {name} — {}/{} ({:.0}%)",
            index + 1,
            bounds.len(),
            range.start as f64 / pcm.len().max(1) as f64 * 100.0
        ));
        let offset = range.start as f64 / ASR_SAMPLE_RATE;
        let part = client
            .transcribe_segments(&pcm[range.clone()], &context, Quality::ACCURATE)
            .await
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        texts.extend(part.into_iter().map(|t| TimedText {
            start: t.start + offset,
            end: t.end + offset,
            text: t.text,
        }));
        // Końcówka poprzedniego kawałka jako `prompt` prawie połowi błędy
        // w nazwach własnych — ten sam trik co na żywo.
        let joined = texts
            .iter()
            .map(|t| t.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let skip = joined.chars().count().saturating_sub(400);
        context = joined.chars().skip(skip).collect();
    }

    let mut turns = Vec::new();
    let mut diarization_note = None;
    if let Some(engine) = diarizer.filter(|_| !texts.is_empty()) {
        progress("Rozpoznaję głosy…".into());
        let wav = std::env::temp_dir().join(format!("cw-import-{}.wav", std::process::id()));
        let tape = crate::diarization::WavFileWriter::create(wav.clone())?;
        tape.append(&pcm, 0.0);
        tape.finish();
        let path = wav.clone();
        match tokio::task::spawn_blocking(move || engine.run(&path, None)).await? {
            Ok(found) => turns = found,
            // Transkrypt bez etykiet jest wciąż wart więcej niż żaden.
            Err(err) => diarization_note = Some(err.to_string()),
        }
        let _ = std::fs::remove_file(wav);
    }
    let labels = SpeakerTurns::label(&texts, &turns, "Osoba", "Nagranie");
    let paragraphs = SpeakerTurns::paragraphs(&texts, &labels, 2.5, 60.0);
    let started_at = std::fs::metadata(file)
        .and_then(|m| m.created().or_else(|_| m.modified()))
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs_f64() * 1000.0)
        .unwrap_or_else(cw_core::time::now_ms);
    let segments = SpeakerTurns::segments(&paragraphs, started_at, "imp");
    let meta = SessionMeta {
        title: file
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default(),
        source: "file".into(),
        url: file.to_string_lossy().into_owned(),
        started_at: Some(started_at),
        ended_at: Some(started_at + seconds * 1000.0),
    };
    Ok(ImportResult {
        segments,
        meta,
        diarization_note,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuts_in_the_quietest_spot() {
        let rate = 16_000;
        // 3 s głośno, cisza w 2,5 s, potem głośno dalej.
        let mut samples = vec![0.5f32; 5 * rate];
        for s in &mut samples[(2.5 * rate as f64) as usize..(2.6 * rate as f64) as usize] {
            *s = 0.0;
        }
        let bounds = chunk_bounds(samples.len(), 3 * rate, 2 * rate, &samples);
        assert_eq!(bounds.len(), 2);
        let cut = bounds[0].end as f64 / rate as f64;
        assert!((2.5..=2.6).contains(&cut), "{cut}");
        assert_eq!(bounds[1].end, samples.len());
    }

    #[test]
    fn short_file_is_one_chunk() {
        let samples = vec![0.1f32; 1000];
        assert_eq!(chunk_bounds(1000, 16_000, 1600, &samples), vec![0..1000]);
    }

    /// `CW_TEST_AUDIO=plik CW_TEST_PORT=8898 cargo test -p call-whisper -- --ignored import_real`
    #[tokio::test]
    #[ignore]
    async fn import_real_file() {
        let file = std::env::var("CW_TEST_AUDIO").expect("CW_TEST_AUDIO");
        let port: u16 = std::env::var("CW_TEST_PORT")
            .unwrap_or("8898".into())
            .parse()
            .unwrap();
        let result = run(Path::new("ffmpeg"), Path::new(&file), port, "pl", None, |m| {
            eprintln!("{m}")
        })
        .await
        .unwrap();
        let text: String = result
            .segments
            .iter()
            .map(|s| s.text.clone())
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!("{} segmentów: {text}", result.segments.len());
        assert!(text.to_lowercase().contains("projekcie"), "{text}");
        assert_eq!(result.segments[0].speaker, "Nagranie");
    }
}
