//! Rozpoznawanie głosów po rozmowie: pyannote (segmentacja) + CAM++
//! (odcisk głosu) przez `sherpa-onnx-offline-speaker-diarization` — to samo
//! co na macOS. Silnik i modele są w instalatorze (vendor/sherpa).

use crate::whisper_server::hide_console;
use cw_core::speaker_turns::{SpeakerTurn, SpeakerTurns};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;

/// Nagranie dźwięku komputera na dysk, w trakcie rozmowy.
pub struct WavFileWriter {
    pub path: PathBuf,
    inner: Mutex<(std::fs::File, usize)>,
}

const SAMPLE_RATE: f64 = 16_000.0;

impl WavFileWriter {
    pub fn create(path: PathBuf) -> std::io::Result<Self> {
        let mut file = std::fs::File::create(&path)?;
        file.write_all(&header(0))?;
        Ok(Self {
            path,
            inner: Mutex::new((file, 0)),
        })
    }

    /// `start_ms`: czas pierwszej próbki od początku nagrania. Loopback
    /// WASAPI milknie przy ciszy w systemie — lukę wypełniamy zerami, żeby
    /// czasy tur zgadzały się z transkryptem. 20 ms tolerancji na drganie.
    pub fn append(&self, samples: &[f32], start_ms: f64) {
        let mut guard = self.inner.lock().unwrap();
        let (file, written) = &mut *guard;
        let expected = (start_ms / 1000.0 * SAMPLE_RATE).round() as usize;
        let gap = expected.saturating_sub(*written);
        if gap > (SAMPLE_RATE / 50.0) as usize {
            let _ = file.write_all(&vec![0u8; gap * 2]);
            *written += gap;
        }
        let mut data = Vec::with_capacity(samples.len() * 2);
        for &s in samples {
            let c = s.clamp(-1.0, 1.0);
            let v = if c < 0.0 { c * 32768.0 } else { c * 32767.0 } as i16;
            data.extend_from_slice(&v.to_le_bytes());
        }
        let _ = file.write_all(&data);
        *written += samples.len();
    }

    /// Dopisuje prawdziwe rozmiary do nagłówka.
    pub fn finish(&self) {
        let mut guard = self.inner.lock().unwrap();
        let (file, written) = &mut *guard;
        let _ = file.seek(SeekFrom::Start(0));
        let _ = file.write_all(&header(*written));
        let _ = file.flush();
    }

    #[cfg(test)]
    pub fn samples(&self) -> usize {
        self.inner.lock().unwrap().1
    }
}

fn header(samples: usize) -> Vec<u8> {
    let data_len = (samples * 2) as u32;
    let mut h = Vec::with_capacity(44);
    h.extend_from_slice(b"RIFF");
    h.extend_from_slice(&(36 + data_len).to_le_bytes());
    h.extend_from_slice(b"WAVEfmt ");
    h.extend_from_slice(&16u32.to_le_bytes());
    h.extend_from_slice(&1u16.to_le_bytes()); // PCM
    h.extend_from_slice(&1u16.to_le_bytes()); // mono
    h.extend_from_slice(&16_000u32.to_le_bytes());
    h.extend_from_slice(&32_000u32.to_le_bytes());
    h.extend_from_slice(&2u16.to_le_bytes());
    h.extend_from_slice(&16u16.to_le_bytes());
    h.extend_from_slice(b"data");
    h.extend_from_slice(&data_len.to_le_bytes());
    h
}

pub struct Engine {
    binary: PathBuf,
    segmentation: PathBuf,
    embedding: PathBuf,
}

impl Engine {
    pub fn locate(vendor: &Path) -> Option<Self> {
        let dir = vendor.join("sherpa");
        let binary = dir.join(if cfg!(windows) {
            "sherpa-onnx-offline-speaker-diarization.exe"
        } else {
            "sherpa-onnx-offline-speaker-diarization"
        });
        let engine = Self {
            binary,
            segmentation: dir.join("segmentation.onnx"),
            embedding: dir.join("embedding.onnx"),
        };
        (engine.binary.is_file() && engine.segmentation.is_file() && engine.embedding.is_file())
            .then_some(engine)
    }

    /// Tury mówców w nagraniu. `speakers` = znana liczba osób; `None` =
    /// niech zdecyduje klastrowanie (próg 0,5 jak na macOS).
    pub fn run(&self, wav: &Path, speakers: Option<usize>) -> anyhow::Result<Vec<SpeakerTurn>> {
        let mut command = Command::new(&self.binary);
        command
            .arg(format!(
                "--segmentation.pyannote-model={}",
                self.segmentation.display()
            ))
            .arg(format!("--embedding.model={}", self.embedding.display()))
            .arg("--min-duration-on=0.2")
            .arg("--min-duration-off=0.5");
        match speakers {
            Some(n) if n > 0 => command.arg(format!("--clustering.num-clusters={n}")),
            _ => command.arg("--clustering.cluster-threshold=0.5"),
        };
        command.arg(wav).stdin(Stdio::null());
        hide_console(&mut command);
        // `output()` czyta oba strumienie naraz, więc gadatliwy stderr
        // (cała konfiguracja) nie zablokuje procesu.
        let output = command.output()?;
        if !output.status.success() {
            let err = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!(
                "Rozpoznawanie głosów nie wyszło: {}",
                err.trim().lines().last().unwrap_or("")
            );
        }
        Ok(SpeakerTurns::parse(&String::from_utf8_lossy(
            &output.stdout,
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fills_gaps_with_silence_and_fixes_header() {
        let path = std::env::temp_dir().join(format!("cw-test-{}.wav", std::process::id()));
        let tape = WavFileWriter::create(path.clone()).unwrap();
        tape.append(&[0.5; 1600], 0.0);
        // Paczka po sekundzie przerwy (loopback milczał).
        tape.append(&[0.5; 1600], 1100.0);
        tape.finish();
        assert_eq!(tape.samples(), 17_600 + 1600);
        let data = std::fs::read(&path).unwrap();
        assert_eq!(data.len(), 44 + (17_600 + 1600) * 2);
        assert_eq!(
            u32::from_le_bytes(data[40..44].try_into().unwrap()) as usize,
            (17_600 + 1600) * 2
        );
        let _ = std::fs::remove_file(path);
    }

    /// `CW_TEST_VENDOR=katalog CW_TEST_AUDIO=plik.wav cargo test -p call-whisper -- --ignored diarize_real`
    #[test]
    #[ignore]
    fn diarize_real_recording() {
        let vendor = PathBuf::from(std::env::var("CW_TEST_VENDOR").expect("CW_TEST_VENDOR"));
        let engine = Engine::locate(&vendor).expect("silnik w vendor/sherpa");
        let turns = engine.run(Path::new(&std::env::var("CW_TEST_AUDIO").unwrap()), None).unwrap();
        let clusters: std::collections::HashSet<_> = turns.iter().map(|t| t.cluster.clone()).collect();
        eprintln!("{} tur, {} głosów", turns.len(), clusters.len());
        assert!(clusters.len() >= 2);
    }
}
