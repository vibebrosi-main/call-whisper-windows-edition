//! Przechwytywanie dźwięku przez WASAPI (cpal): komputer = loopback
//! domyślnego wyjścia, mikrofon = wybrane wejście. Oba lecą jako 16 kHz mono.

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use serde::Serialize;
use std::sync::mpsc;
use std::thread::JoinHandle;

pub const ASR_SAMPLE_RATE: f64 = 16_000.0;

/// Skąd przyszedł dźwięk. Rozdzielenie źródeł daje za darmo pewny podział
/// „ja" vs „oni": mikrofon to Ty, dźwięk komputera to zdalni uczestnicy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum AudioSource {
    System,
    Microphone,
}

impl AudioSource {
    pub fn raw(self) -> &'static str {
        match self {
            AudioSource::System => "system",
            AudioSource::Microphone => "microphone",
        }
    }
}

pub struct PcmChunk {
    pub source: AudioSource,
    pub samples: Vec<f32>,
    /// Czas pierwszej próbki względem startu sesji, w ms.
    pub start_ms: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct InputDevice {
    pub id: String,
    pub name: String,
}

/// Mikrofony do wyboru. W cpal identyfikatorem urządzenia jest jego nazwa.
pub fn input_devices() -> Vec<InputDevice> {
    let host = cpal::default_host();
    host.input_devices()
        .map(|devices| {
            devices
                .filter_map(|d| d.name().ok())
                .map(|name| InputDevice { id: name.clone(), name })
                .collect()
        })
        .unwrap_or_default()
}

pub fn default_input_device() -> Option<InputDevice> {
    cpal::default_host()
        .default_input_device()
        .and_then(|d| d.name().ok())
        .map(|name| InputDevice { id: name.clone(), name })
}

/// Działające przechwytywanie. Strumień cpal żyje w osobnym wątku, bo nie
/// na każdej platformie jest `Send`; zatrzymanie = zamknięcie kanału.
pub struct Capture {
    stop: Option<mpsc::Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl Capture {
    /// `device_id` pusty albo nieznany (urządzenie odłączone) = domyślne.
    pub fn start(
        source: AudioSource,
        device_id: &str,
        session_start_ms: f64,
        on_chunk: impl Fn(PcmChunk) + Send + 'static,
        on_error: impl Fn(String) + Send + 'static,
    ) -> anyhow::Result<Self> {
        let device_id = device_id.to_string();
        let (stop_tx, stop_rx) = mpsc::channel::<()>();
        let (ready_tx, ready_rx) = mpsc::channel::<anyhow::Result<()>>();

        let thread = std::thread::spawn(move || {
            let stream = match build_stream(source, &device_id, session_start_ms, on_chunk, on_error) {
                Ok(stream) => stream,
                Err(err) => {
                    let _ = ready_tx.send(Err(err));
                    return;
                }
            };
            let _ = ready_tx.send(Ok(()));
            // Czekamy na sygnał stopu albo zamknięcie kanału.
            let _ = stop_rx.recv();
            drop(stream);
        });

        ready_rx.recv().map_err(|_| anyhow::anyhow!("wątek audio padł przy starcie"))??;
        Ok(Self { stop: Some(stop_tx), thread: Some(thread) })
    }

    pub fn stop(&mut self) {
        self.stop.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for Capture {
    fn drop(&mut self) {
        self.stop();
    }
}

fn build_stream(
    source: AudioSource,
    device_id: &str,
    session_start_ms: f64,
    on_chunk: impl Fn(PcmChunk) + Send + 'static,
    on_error: impl Fn(String) + Send + 'static,
) -> anyhow::Result<cpal::Stream> {
    let host = cpal::default_host();
    let (device, config) = match source {
        // Loopback WASAPI: strumień wejściowy zbudowany na urządzeniu
        // wyjściowym oddaje to, co gra komputer. Bez zgód i bez sterowników.
        AudioSource::System => {
            let device = host
                .default_output_device()
                .ok_or_else(|| anyhow::anyhow!("Brak urządzenia wyjściowego audio."))?;
            let config = device.default_output_config()?;
            (device, config)
        }
        AudioSource::Microphone => {
            let chosen = (!device_id.is_empty())
                .then(|| {
                    host.input_devices()
                        .ok()?
                        .find(|d| d.name().map(|n| n == device_id).unwrap_or(false))
                })
                .flatten();
            let device = chosen
                .or_else(|| host.default_input_device())
                .ok_or_else(|| anyhow::anyhow!("Brak urządzenia wejściowego audio."))?;
            let config = device.default_input_config()?;
            (device, config)
        }
    };

    let channels = config.channels() as usize;
    let rate = config.sample_rate().0 as f64;
    let format = config.sample_format();
    let config: cpal::StreamConfig = config.into();

    let mut resampler = Resampler::new(rate, ASR_SAMPLE_RATE);
    let mut emitted: u64 = 0;
    let mut base_ms: Option<f64> = None;
    let mut mono: Vec<f32> = Vec::new();

    // Czas z liczby próbek (zegar urządzenia), zakotwiczony w chwili
    // pierwszej paczki — ta sama zasada co na macOS.
    let mut push = move |mono: &[f32]| {
        let samples = resampler.process(mono);
        if samples.is_empty() {
            return;
        }
        let base = *base_ms.get_or_insert_with(|| (cw_now_ms() - session_start_ms).max(0.0));
        let start_ms = base + emitted as f64 / ASR_SAMPLE_RATE * 1000.0;
        emitted += samples.len() as u64;
        on_chunk(PcmChunk { source, samples, start_ms });
    };

    let err_fn = move |err: cpal::StreamError| on_error(err.to_string());

    let stream = match format {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &config,
            move |data: &[f32], _: &_| {
                downmix(data, channels, &mut mono, |s| s);
                push(&mono);
            },
            err_fn,
            None,
        )?,
        cpal::SampleFormat::I16 => device.build_input_stream(
            &config,
            move |data: &[i16], _: &_| {
                downmix(data, channels, &mut mono, |s| s as f32 / 32768.0);
                push(&mono);
            },
            err_fn,
            None,
        )?,
        cpal::SampleFormat::I32 => device.build_input_stream(
            &config,
            move |data: &[i32], _: &_| {
                downmix(data, channels, &mut mono, |s| s as f32 / 2_147_483_648.0);
                push(&mono);
            },
            err_fn,
            None,
        )?,
        other => anyhow::bail!("Nieobsługiwany format próbek: {other:?}"),
    };
    stream.play()?;
    Ok(stream)
}

fn downmix<T: Copy>(data: &[T], channels: usize, out: &mut Vec<f32>, to_f32: impl Fn(T) -> f32) {
    out.clear();
    let channels = channels.max(1);
    out.extend(data.chunks(channels).map(|frame| {
        frame.iter().map(|&s| to_f32(s)).sum::<f32>() / frame.len() as f32
    }));
}

fn cw_now_ms() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64() * 1000.0)
        .unwrap_or(0.0)
}

/// Strumieniowy resampler: każda próbka wyjściowa to średnia próbek wejścia
/// z jej przedziału czasu. Średnia działa jak filtr dolnoprzepustowy, więc
/// 48 kHz -> 16 kHz nie zostawia aliasów, które psułyby whispera.
pub struct Resampler {
    step: f64,
    pos: f64,
    acc: f32,
    count: u32,
}

impl Resampler {
    pub fn new(from: f64, to: f64) -> Self {
        Self { step: from / to, pos: 0.0, acc: 0.0, count: 0 }
    }

    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity((input.len() as f64 / self.step) as usize + 1);
        for &sample in input {
            self.acc += sample;
            self.count += 1;
            self.pos += 1.0;
            if self.pos >= self.step {
                self.pos -= self.step;
                out.push(self.acc / self.count as f32);
                self.acc = 0.0;
                self.count = 0;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resampler_keeps_rate() {
        let mut r = Resampler::new(48_000.0, 16_000.0);
        let total: usize = (0..100).map(|_| r.process(&[0.5; 480]).len()).sum();
        assert_eq!(total, 16_000);
    }

    #[test]
    fn resampler_handles_fractional_ratio() {
        let mut r = Resampler::new(44_100.0, 16_000.0);
        let total: usize = (0..100).map(|_| r.process(&[0.1; 441]).len()).sum();
        assert!((total as i64 - 16_000).abs() <= 1, "{total}");
    }

    #[test]
    fn downmix_averages_channels() {
        let mut out = Vec::new();
        downmix(&[1.0f32, 0.0, 0.5, 0.5], 2, &mut out, |s| s);
        assert_eq!(out, vec![0.5, 0.5]);
    }
}
