//! Kodowanie PCM do WAV.
//!
//! `whisper-server` przyjmuje pliki, nie strumienie, więc każdą wypowiedź
//! trzeba opakować w kontener. 16-bit PCM mono to najprostszy format, który
//! rozumie każdy dekoder.
//!
//! Port z `Wav.swift` (a ten z `extension/src/adapters/audio/wav.js`).

pub struct Wav;

const HEADER_BYTES: usize = 44;

/// Float32 [-1, 1] -> int16 z obcięciem zakresu.
fn to_i16(sample: f32) -> i16 {
    // Kolejność porównań jak Swift.min/Swift.max: NaN ląduje na 1.
    let upper = if sample < 1.0 { sample } else { 1.0 };
    let clamped = if upper >= -1.0 { upper } else { -1.0 };
    let scaled = if clamped < 0.0 { clamped as f64 * 32768.0 } else { clamped as f64 * 32767.0 };
    scaled.trunc() as i16
}

impl Wav {
    /// Swift: domyślnie `sample_rate = 16_000`, `channels = 1`.
    pub fn encode(samples: &[f32], sample_rate: u32, channels: u16) -> Vec<u8> {
        let bytes_per_sample: u32 = 2;
        let data_bytes = samples.len() as u32 * bytes_per_sample;
        let mut data = Vec::with_capacity(HEADER_BYTES + data_bytes as usize);

        data.extend_from_slice(b"RIFF");
        data.extend_from_slice(&(36 + data_bytes).to_le_bytes()); // rozmiar pliku - 8
        data.extend_from_slice(b"WAVE");

        data.extend_from_slice(b"fmt ");
        data.extend_from_slice(&16u32.to_le_bytes()); // długość bloku fmt
        data.extend_from_slice(&1u16.to_le_bytes()); // 1 = PCM bez kompresji
        data.extend_from_slice(&channels.to_le_bytes());
        data.extend_from_slice(&sample_rate.to_le_bytes());
        data.extend_from_slice(&(sample_rate * channels as u32 * bytes_per_sample).to_le_bytes()); // bajtów na sekundę
        data.extend_from_slice(&(channels * bytes_per_sample as u16).to_le_bytes()); // wyrównanie bloku
        data.extend_from_slice(&(8 * bytes_per_sample as u16).to_le_bytes());

        data.extend_from_slice(b"data");
        data.extend_from_slice(&data_bytes.to_le_bytes());

        for &sample in samples {
            data.extend_from_slice(&to_i16(sample).to_le_bytes());
        }
        data
    }

    /// Długość audio w ms dla danej liczby próbek. Swift: domyślnie 16 kHz.
    pub fn duration_ms(sample_count: usize, sample_rate: f64) -> f64 {
        sample_count as f64 / sample_rate * 1000.0
    }
}
