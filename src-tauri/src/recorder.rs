//! Spina wszystko w całość: dźwięk -> diaryzacja + ASR -> transkrypt ->
//! interfejs. Port `Recorder.swift`. Interfejs dostaje pełny stan zdarzeniem
//! `state` przy każdej zmianie.

use crate::audio::{AudioSource, Capture};
use crate::model_downloader;
use crate::pipeline::{self, Event, Pipeline};
use crate::settings::{AssistantBackend, Settings};
use crate::whisper_client::WhisperClient;
use crate::whisper_server::{self, WhisperServer};
use cw_core::assistant::{AssistantItem, AssistantSession};
use cw_core::markdown::{Markdown, MarkdownOptions, Session, SessionMeta};
use cw_core::time::{now_ms, TimeFormat};
use cw_core::transcript_store::{Segment, TranscriptStore};
use cw_core::vocabulary::Vocabulary;
use serde::Serialize;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tauri::{AppHandle, Emitter};
use tokio::sync::mpsc;
use tokio::task::JoinHandle;

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AppState {
    pub segments: Vec<Segment>,
    pub answers: Vec<AssistantItem>,
    pub is_running: bool,
    pub is_processing: bool,
    pub status: String,
    pub last_error: Option<String>,
    pub speaker_count: usize,
    pub started_at: Option<f64>,
    pub title: Option<String>,
    pub overlay_visible: bool,
    pub assistant_label: String,
    pub meeting: Option<String>,
    pub update: Option<String>,
    pub readiness_ok: bool,
    pub file_base: String,
    pub media_extensions: Vec<&'static str>,
}

#[derive(Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Check {
    pub id: &'static str,
    pub title: String,
    pub ok: bool,
    pub detail: String,
    pub action: Option<String>,
    pub action_label: Option<String>,
}

struct Inner {
    store: TranscriptStore,
    session: AssistantSession,
    /// Metadane sesji z importu; `None` = rozmowa nagrana na żywo.
    imported_meta: Option<SessionMeta>,
    is_running: bool,
    is_processing: bool,
    status: String,
    last_error: Option<String>,
    started_at: Option<f64>,
    pipelines: Vec<Pipeline>,
    captures: Vec<Capture>,
    sources: Vec<AudioSource>,
    tasks: Vec<JoinHandle<()>>,
    processing: Option<JoinHandle<()>>,
    overlay_visible: bool,
    meeting: Option<String>,
    update: Option<String>,
}

pub struct Recorder {
    app: AppHandle,
    pub vendor: PathBuf,
    inner: Mutex<Inner>,
    pub settings: Mutex<Settings>,
    server: tokio::sync::Mutex<WhisperServer>,
}

pub const MEDIA_EXTENSIONS: &[&str] = &["mp4", "mov", "m4a", "mp3", "wav", "aac", "flac", "ogg", "webm", "mkv", "m4v"];

impl Recorder {
    pub fn new(app: AppHandle, vendor: PathBuf) -> Arc<Self> {
        Arc::new(Self {
            app,
            server: tokio::sync::Mutex::new(WhisperServer::new(vendor.clone())),
            vendor,
            settings: Mutex::new(Settings::load()),
            inner: Mutex::new(Inner {
                store: TranscriptStore::new(now_ms()),
                session: AssistantSession::new(20),
                imported_meta: None,
                is_running: false,
                is_processing: false,
                status: "Gotowy".into(),
                last_error: None,
                started_at: None,
                pipelines: Vec::new(),
                captures: Vec::new(),
                sources: Vec::new(),
                tasks: Vec::new(),
                processing: None,
                overlay_visible: false,
                meeting: None,
                update: None,
            }),
        })
    }

    pub fn settings(&self) -> Settings {
        self.settings.lock().unwrap().clone()
    }

    pub fn is_busy(&self) -> bool {
        let inner = self.inner.lock().unwrap();
        inner.is_running || inner.is_processing
    }

    // --- stan dla interfejsu ---

    pub fn state(&self) -> AppState {
        let settings = self.settings();
        let inner = self.inner.lock().unwrap();
        let file_base = inner
            .imported_meta
            .as_ref()
            .map(|m| m.title.clone())
            .filter(|t| !t.is_empty())
            .unwrap_or_else(|| format!("rozmowa-{}", TimeFormat::filename_stamp(inner.started_at.unwrap_or_else(now_ms))));
        AppState {
            segments: inner.store.segments(),
            answers: inner.session.all(),
            is_running: inner.is_running,
            is_processing: inner.is_processing,
            status: inner.status.clone(),
            last_error: inner.last_error.clone(),
            speaker_count: inner.pipelines.iter().map(|p| p.speaker_count()).sum(),
            started_at: inner.started_at,
            title: inner.imported_meta.as_ref().map(|m| m.title.clone()),
            overlay_visible: inner.overlay_visible,
            assistant_label: assistant_label(&settings),
            meeting: inner.meeting.clone(),
            update: inner.update.clone(),
            readiness_ok: self.readiness().iter().all(|c| c.ok),
            file_base,
            media_extensions: MEDIA_EXTENSIONS.to_vec(),
        }
    }

    pub fn publish(&self) {
        let _ = self.app.emit("state", self.state());
    }

    fn set_status(&self, status: impl Into<String>) {
        self.inner.lock().unwrap().status = status.into();
        self.publish();
    }

    fn set_error(&self, error: impl Into<String>) {
        self.inner.lock().unwrap().last_error = Some(error.into());
        self.publish();
    }

    pub fn set_update(&self, update: Option<String>) {
        self.inner.lock().unwrap().update = update;
        self.publish();
    }

    pub fn set_overlay_visible(&self, visible: bool) {
        self.inner.lock().unwrap().overlay_visible = visible;
        self.publish();
    }

    pub fn set_meeting(&self, meeting: Option<String>) {
        self.inner.lock().unwrap().meeting = meeting;
        self.publish();
    }

    // --- gotowość ---

    /// Czego aplikacja potrzebuje, żeby w ogóle zadziałać — i co z tego
    /// potrafi załatwić sama. Na Windows nie ma zgody na nagrywanie ekranu:
    /// loopback WASAPI jej nie wymaga.
    pub fn readiness(&self) -> Vec<Check> {
        let settings = self.settings();
        let mut checks = Vec::new();
        let has_engine = self.server.try_lock().map(|s| s.binary_available()).unwrap_or(true);
        checks.push(Check {
            id: "whisper",
            title: "Silnik mowy (whisper.cpp)".into(),
            ok: has_engine,
            detail: if has_engine { "Wbudowany (GPU przez Vulkan, zapasowo CPU).".into() } else { "Brak w instalacji — zainstaluj aplikację ponownie.".into() },
            action: None,
            action_label: None,
        });
        let model = settings.whisper_model.clone();
        let has_model = model_downloader::is_installed(&model);
        let size = model_downloader::KNOWN.iter().find(|m| m.id == model).map(|m| m.bytes / 1_000_000).unwrap_or(0);
        checks.push(Check {
            id: "model",
            title: format!("Model mowy ({model})"),
            ok: has_model,
            detail: if has_model { "Pobrany.".into() } else { format!("Pobierze się sam przy pierwszym nasłuchu ({size} MB) albo teraz.") },
            action: (!has_model).then(|| "downloadModel".into()),
            action_label: (!has_model).then(|| "Pobierz".into()),
        });
        let mics = crate::audio::input_devices();
        checks.push(Check {
            id: "mic",
            title: "Mikrofon".into(),
            ok: !settings.use_microphone || !mics.is_empty(),
            detail: if mics.is_empty() { "Nie wykryto żadnego mikrofonu.".into() } else { format!("Dostępne: {}", mics.len()) },
            action: None,
            action_label: None,
        });
        if settings.assistant_enabled && settings.assistant_backend == AssistantBackend::Api {
            checks.push(Check {
                id: "key",
                title: "Klucz API".into(),
                ok: !settings.api_key.is_empty(),
                detail: "Potrzebny do podpowiedzi przez API. Wpisz go w Ustawieniach.".into(),
                action: None,
                action_label: None,
            });
        }
        checks
    }

    pub async fn readiness_action(self: &Arc<Self>, action: &str) {
        if action == "downloadModel" {
            let model = self.settings().whisper_model;
            self.inner.lock().unwrap().is_processing = true;
            let _ = self.download_model(&model).await;
            self.inner.lock().unwrap().is_processing = false;
            self.set_status("Gotowy");
        }
    }

    async fn download_model(self: &Arc<Self>, model: &str) -> bool {
        self.set_status(format!("Pobieram model {model}…"));
        let this = self.clone();
        let name = model.to_string();
        let result = model_downloader::download(model, move |p| {
            this.set_status(format!(
                "Pobieram model {name} — {:.0}% ({:.0}/{:.0} MB)",
                p.fraction * 100.0,
                p.received_bytes as f64 / 1e6,
                p.total_bytes as f64 / 1e6
            ));
        })
        .await;
        match result {
            Ok(()) => true,
            Err(err) => {
                self.set_error(format!("Nie udało się pobrać modelu: {err}"));
                self.set_status("Gotowy");
                false
            }
        }
    }

    // --- cykl życia ---

    /// Podnosi `whisper-server`, jeśli trzeba. `false`, gdy się nie da.
    async fn prepare_whisper(self: &Arc<Self>) -> bool {
        let settings = self.settings();
        if !model_downloader::is_installed(&settings.whisper_model) && !self.download_model(&settings.whisper_model).await {
            return false;
        }
        let client = WhisperClient::new(settings.whisper_port, &settings.language_code(), 15);
        if client.health().await {
            self.set_status("whisper-server działa");
            return true;
        }
        if !settings.auto_start_whisper {
            self.set_error("whisper-server nie działa. Włącz automatyczne uruchamianie w Ustawieniach.");
            return false;
        }
        self.set_status(format!("Uruchamiam whisper-server ({})…", settings.whisper_model));
        let config = whisper_server::Config {
            model: settings.whisper_model.clone(),
            port: settings.whisper_port,
            language: settings.language_code(),
            threads: std::thread::available_parallelism().map(|n| n.get().min(8)).unwrap_or(4),
        };
        match self.server.lock().await.ensure_running(&config).await {
            Ok(_) => true,
            Err(err) => {
                self.set_error(err.to_string());
                self.set_status("Gotowy");
                false
            }
        }
    }

    pub async fn start(self: &Arc<Self>) {
        let settings = self.settings();
        {
            let mut inner = self.inner.lock().unwrap();
            if inner.is_running || inner.is_processing {
                return;
            }
            if !settings.use_system_audio && !settings.use_microphone {
                inner.last_error = Some("Wybierz źródło dźwięku: komputer albo mikrofon".into());
                drop(inner);
                self.publish();
                return;
            }
            inner.last_error = None;
            inner.imported_meta = None;
            inner.is_processing = true;
        }
        self.publish();

        let ready = self.prepare_whisper().await;
        if !ready {
            self.inner.lock().unwrap().is_processing = false;
            self.publish();
            return;
        }

        let now = now_ms();
        let (events_tx, events_rx) = mpsc::unbounded_channel();
        let mut sources = Vec::new();
        if settings.use_system_audio {
            sources.push(AudioSource::System);
        }
        if settings.use_microphone {
            sources.push(AudioSource::Microphone);
        }
        let project_context = std::fs::read_to_string(&settings.project_context_path).unwrap_or_default();
        let vocabulary = Vocabulary::merge(&settings.whisper_vocabulary, &project_context, Vocabulary::MAX_CHARS);

        let mut pipelines = Vec::new();
        let mut captures = Vec::new();
        let mut failure = None;
        for &source in &sources {
            let pipeline = Pipeline::start(
                pipeline::Config {
                    source,
                    whisper_port: settings.whisper_port,
                    language_code: settings.language_code(),
                    identify_speakers: settings.identify_speakers,
                    vocabulary: vocabulary.clone(),
                },
                events_tx.clone(),
            );
            let errors = events_tx.clone();
            let device = if source == AudioSource::Microphone { settings.mic_device_id.as_str() } else { "" };
            match Capture::start(source, device, now, pipeline.sender(), move |err| {
                let _ = errors.send(Event::Error(format!("Przechwytywanie przerwane: {err}")));
            }) {
                Ok(capture) => captures.push(capture),
                Err(err) => {
                    failure = Some(err.to_string());
                    pipelines.push(pipeline);
                    break;
                }
            }
            pipelines.push(pipeline);
        }

        if let Some(err) = failure {
            for p in pipelines {
                p.cancel();
            }
            let mut inner = self.inner.lock().unwrap();
            inner.is_processing = false;
            inner.last_error = Some(err);
            inner.status = "Gotowy".into();
            drop(inner);
            self.publish();
            return;
        }

        let heard = sources
            .iter()
            .map(|s| if *s == AudioSource::System { "system" } else { "mikrofon" })
            .collect::<Vec<_>>()
            .join(" + ");
        {
            let mut inner = self.inner.lock().unwrap();
            inner.store = TranscriptStore::new(now);
            inner.session = AssistantSession::new(20);
            inner.started_at = Some(now);
            inner.pipelines = pipelines;
            inner.captures = captures;
            inner.sources = sources;
            inner.is_running = true;
            inner.is_processing = false;
            inner.status = format!("Słucham: {heard} · whisper {} · {}", settings.whisper_model, settings.language_code());
            inner.tasks.push(self.spawn_events(events_rx));
            inner.tasks.push(self.spawn_ticker());
        }
        self.publish();
    }

    /// Aktualizacje z łańcuchów trafiają do `TranscriptStore`.
    fn spawn_events(self: &Arc<Self>, mut rx: mpsc::UnboundedReceiver<Event>) -> JoinHandle<()> {
        let this = self.clone();
        tokio::spawn(async move {
            while let Some(event) = rx.recv().await {
                match event {
                    Event::Update(update) => this.apply(update),
                    Event::Error(err) => {
                        let mut inner = this.inner.lock().unwrap();
                        // Ten sam błąd potrafi się powtórzyć co rundę — w interfejsie raz.
                        if inner.last_error.as_deref() != Some(err.as_str()) {
                            inner.last_error = Some(err);
                        }
                    }
                }
                this.publish();
            }
        })
    }

    /// Co 250 ms odświeżamy interfejs (liczba mówców, czasy).
    fn spawn_ticker(self: &Arc<Self>) -> JoinHandle<()> {
        let this = self.clone();
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(250));
            loop {
                interval.tick().await;
                this.publish();
            }
        })
    }

    fn apply(&self, update: pipeline::Update) {
        let mut inner = self.inner.lock().unwrap();
        let now = now_ms();
        if update.discard {
            // Wypowiedź bez tekstu — usuwamy segment zamiast pustego wiersza.
            inner.store.discard(&update.key);
            inner.store.seal(&update.key, now);
            return;
        }
        // Cokolwiek dojechało, znaczy że łańcuch działa.
        inner.last_error = None;
        let at = inner.started_at.unwrap_or(now) + update.start_ms;
        // `replace: true`: whisper podaje pełną treść przy każdej rundzie.
        inner.store.upsert(&update.key, Some(&update.speaker), &update.text, at, true);
        if update.is_final {
            // Pieczętujemy klucz: ten sam wynik potrafi przyjść ponownie.
            inner.store.seal(&update.key, now);
        }
    }

    /// Najpierw gasimy stan widoczny w interfejsie, potem sprzątamy.
    pub async fn stop(self: &Arc<Self>) {
        let (pipelines, captures, tasks) = {
            let mut inner = self.inner.lock().unwrap();
            if !inner.is_running && inner.pipelines.is_empty() {
                return;
            }
            inner.is_running = false;
            inner.status = "Zatrzymuję…".into();
            (
                std::mem::take(&mut inner.pipelines),
                std::mem::take(&mut inner.captures),
                std::mem::take(&mut inner.tasks),
            )
        };
        self.publish();
        drop(captures);
        for p in pipelines {
            p.cancel();
        }
        for t in tasks {
            t.abort();
        }
        // Zatrzymujemy tylko serwer, który sami podnieśliśmy.
        self.server.lock().await.stop();
        {
            let mut inner = self.inner.lock().unwrap();
            inner.store.finalize_all(now_ms());
            inner.status = if inner.store.is_empty() { "Gotowy".into() } else { "Zatrzymane".into() };
        }
        self.publish();
    }

    pub fn cancel_processing(&self) {
        if let Some(task) = self.inner.lock().unwrap().processing.take() {
            task.abort();
        }
        let mut inner = self.inner.lock().unwrap();
        inner.is_processing = false;
        inner.status = "Przerwano".into();
        drop(inner);
        self.publish();
    }

    // --- eksport ---

    pub fn segments(&self) -> Vec<Segment> {
        self.inner.lock().unwrap().store.segments()
    }

    pub fn chat_text(&self, ids: Option<Vec<String>>) -> String {
        let segments = self.segments();
        let chosen: Vec<Segment> = match ids {
            Some(ids) => segments.into_iter().filter(|s| ids.contains(&s.id)).collect(),
            None => segments,
        };
        Markdown::chat_text(&chosen)
    }

    pub fn markdown(&self, absolute_timestamps: Option<bool>) -> String {
        let settings = self.settings();
        let inner = self.inner.lock().unwrap();
        let segments = inner.store.segments();
        let source = match inner.sources.as_slice() {
            [AudioSource::Microphone] => "microphone",
            [_, _] => "mixed",
            _ => "system-audio",
        };
        let meta = inner.imported_meta.clone().unwrap_or(SessionMeta {
            title: settings.title.clone(),
            source: source.into(),
            url: String::new(),
            started_at: inner.started_at,
            ended_at: segments.last().map(|s| s.ended_at),
        });
        let options = MarkdownOptions {
            locale: settings.markdown_locale.clone(),
            absolute_timestamps: absolute_timestamps.unwrap_or(settings.absolute_timestamps),
            ..MarkdownOptions::default()
        };
        Markdown::render(&Session { meta, segments }, &options)
    }

    pub fn json(&self) -> String {
        let inner = self.inner.lock().unwrap();
        let payload = serde_json::json!({
            "startedAt": inner.started_at.unwrap_or_else(now_ms),
            "segments": inner.store.segments(),
        });
        serde_json::to_string_pretty(&payload).unwrap_or_default()
    }
}

/// Skąd faktycznie idą podpowiedzi.
pub fn assistant_label(settings: &Settings) -> String {
    if !settings.assistant_enabled {
        return "wyłączone".into();
    }
    match settings.assistant_backend {
        AssistantBackend::ClaudeCode => {
            let model = if settings.claude_model.is_empty() { "domyślny" } else { &settings.claude_model };
            format!("Claude Code · {model}")
        }
        AssistantBackend::Api => {
            if settings.model_id.is_empty() { "API".into() } else { settings.model_id.clone() }
        }
    }
}
