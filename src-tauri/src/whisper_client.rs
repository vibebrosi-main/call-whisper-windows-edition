//! Klient lokalnego whisper.cpp (`whisper-server`).
//!
//! Największą pojedynczą poprawę jakości daje `prompt` (kontekst rozmowy),
//! nie model ani beam search: zmierzone 23,3 % -> 13,3 % WER kosztem 33 ms.

use cw_core::speaker_turns::TimedText;
use cw_core::text::Text;
use cw_core::wav::Wav;
use std::time::Duration;

/// Ustawienia dekodera. Dwa profile, bo służą do czego innego.
#[derive(Clone, Copy, Debug)]
pub struct Quality {
    /// Beam search zamiast zachłannego dekodowania. `None` = zachłannie.
    pub beam_size: Option<u32>,
    /// Odsiewa `[Muzyka]`, `*w tle*` i resztę znaczników nie-mowy.
    pub suppress_non_speech: bool,
}

impl Quality {
    /// Rundy przyrostowe: tekst ma się pojawić szybko, i tak go podmienimy.
    pub const FAST: Quality = Quality { beam_size: None, suppress_non_speech: true };
    /// Wersja ostateczna: ta zostaje w notatce.
    pub const ACCURATE: Quality = Quality { beam_size: Some(5), suppress_non_speech: true };
}

#[derive(Debug)]
pub enum WhisperError {
    Offline(String),
    Http(u16),
}

impl std::fmt::Display for WhisperError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WhisperError::Offline(url) => write!(f, "Nie mogę połączyć się z whisper-server ({url})."),
            WhisperError::Http(status) => write!(f, "whisper-server zwrócił {status}."),
        }
    }
}

impl std::error::Error for WhisperError {}

#[derive(Clone)]
pub struct WhisperClient {
    endpoint: String,
    language: String,
    http: reqwest::Client,
}

impl WhisperClient {
    /// `timeout_s`: 15 s dla rozmowy na żywo, więcej dla importu nagrań,
    /// który wysyła kilkuminutowe kawałki.
    pub fn new(port: u16, language: &str, timeout_s: u64) -> Self {
        Self {
            endpoint: format!("http://127.0.0.1:{port}"),
            language: language.to_string(),
            http: reqwest::Client::builder()
                .timeout(Duration::from_secs(timeout_s))
                .no_proxy()
                .build()
                .expect("reqwest client"),
        }
    }

    pub async fn health(&self) -> bool {
        matches!(
            self.http.get(&self.endpoint).timeout(Duration::from_millis(1500)).send().await,
            Ok(r) if r.status().as_u16() < 500
        )
    }

    /// `pcm`: 16 kHz mono. `context` trafia do `prompt` whispera.
    pub async fn transcribe(&self, pcm: &[f32], context: &str, quality: Quality) -> Result<String, WhisperError> {
        let json = self.inference(pcm, context, quality, "json").await?;
        Ok(Text::clean_whisper(json["text"].as_str().unwrap_or("")))
    }

    /// Segmenty z czasami (`verbose_json`) — do importu nagrań.
    pub async fn transcribe_segments(&self, pcm: &[f32], context: &str, quality: Quality) -> Result<Vec<TimedText>, WhisperError> {
        let json = self.inference(pcm, context, quality, "verbose_json").await?;
        let mut out: Vec<TimedText> = Vec::new();
        let mut raw: Vec<String> = Vec::new();
        for segment in json["segments"].as_array().into_iter().flatten() {
            let (Some(start), Some(end)) = (segment["start"].as_f64(), segment["end"].as_f64()) else { continue };
            let text = segment["text"].as_str().unwrap_or("");
            // Segment bez spacji na początku to ciąg dalszy słowa rozciętego
            // na granicy poprzedniego — doklejamy bez przerwy.
            match (text.chars().next(), out.last_mut()) {
                (Some(first), Some(last)) if !first.is_whitespace() => {
                    raw.last_mut().unwrap().push_str(text);
                    last.end = end;
                }
                _ => {
                    out.push(TimedText { start, end, text: String::new() });
                    raw.push(text.to_string());
                }
            }
        }
        for (segment, text) in out.iter_mut().zip(raw) {
            segment.text = Text::clean_whisper(&text);
        }
        Ok(out.into_iter().filter(|s| !s.text.is_empty()).collect())
    }

    async fn inference(&self, pcm: &[f32], context: &str, quality: Quality, format: &str) -> Result<serde_json::Value, WhisperError> {
        let wav = Wav::encode(pcm, 16_000, 1);
        let file = reqwest::multipart::Part::bytes(wav)
            .file_name("chunk.wav")
            .mime_str("audio/wav")
            .expect("mime");
        let mut form = reqwest::multipart::Form::new()
            .part("file", file)
            .text("language", self.language.clone())
            .text("response_format", format.to_string())
            .text("temperature", "0")
            // Bez tego whisper.cpp dokleja halucynacje na ciszy.
            .text("no_speech_thold", "0.6")
            // Inaczej co trzecie długie słowo jest rozbite („odpow iedzialny").
            .text("split_on_word", "true");
        if quality.suppress_non_speech {
            form = form.text("suppress_nst", "true");
        }
        if let Some(beam) = quality.beam_size {
            form = form.text("beam_size", beam.to_string());
        }
        if !context.is_empty() {
            form = form.text("prompt", context.to_string());
        }

        let response = self
            .http
            .post(format!("{}/inference", self.endpoint))
            .multipart(form)
            .send()
            .await
            .map_err(|_| WhisperError::Offline(self.endpoint.clone()))?;
        if response.status() != 200 {
            return Err(WhisperError::Http(response.status().as_u16()));
        }
        response.json().await.map_err(|_| WhisperError::Offline(self.endpoint.clone()))
    }
}
