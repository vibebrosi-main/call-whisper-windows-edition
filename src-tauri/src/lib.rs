mod audio;
mod model_downloader;
mod pipeline;
mod recorder;
mod settings;
mod updater;
mod whisper_client;
mod whisper_server;

use recorder::{Check, Recorder};
use serde::Serialize;
use settings::Settings;
use std::sync::Arc;
use tauri::{Emitter, Manager, State};

type Rec<'a> = State<'a, Arc<Recorder>>;

/// Funkcje, które dojdą w kolejnych etapach portu z macOS.
const NOT_YET: &str = "Ta funkcja dojdzie w następnej wersji wydania na Windows.";

#[tauri::command]
fn get_state(rec: Rec) -> recorder::AppState {
    rec.state()
}

#[tauri::command]
fn get_settings(rec: Rec) -> Settings {
    rec.settings()
}

#[tauri::command]
fn save_settings(app: tauri::AppHandle, rec: Rec, settings: Settings) -> Result<Settings, String> {
    settings.save().map_err(|e| e.to_string())?;
    *rec.settings.lock().unwrap() = settings.clone();
    let _ = app.emit("settings", &settings);
    rec.publish();
    Ok(settings)
}

#[derive(Serialize)]
struct Mics {
    devices: Vec<audio::InputDevice>,
    default: Option<audio::InputDevice>,
}

#[tauri::command]
fn list_mics() -> Mics {
    Mics { devices: audio::input_devices(), default: audio::default_input_device() }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct ModelInfo {
    id: &'static str,
    note: &'static str,
    installed: bool,
}

#[tauri::command]
fn models() -> Vec<ModelInfo> {
    model_downloader::KNOWN
        .iter()
        .map(|m| ModelInfo { id: m.id, note: m.note, installed: model_downloader::is_installed(m.id) })
        .collect()
}

#[tauri::command]
async fn start(rec: Rec<'_>) -> Result<(), String> {
    rec.inner().start().await;
    Ok(())
}

#[tauri::command]
async fn stop(rec: Rec<'_>) -> Result<(), String> {
    rec.inner().stop().await;
    Ok(())
}

#[tauri::command]
fn cancel_processing(rec: Rec) {
    rec.cancel_processing();
}

#[tauri::command]
fn chat_text(rec: Rec, ids: Option<Vec<String>>) -> String {
    rec.chat_text(ids)
}

#[tauri::command]
fn markdown(rec: Rec) -> String {
    rec.markdown(None)
}

#[tauri::command]
fn json_export(rec: Rec) -> String {
    rec.json()
}

#[tauri::command]
fn write_file(path: String, contents: String) -> Result<(), String> {
    std::fs::write(path, contents).map_err(|e| e.to_string())
}

#[tauri::command]
fn readiness(rec: Rec) -> Vec<Check> {
    rec.readiness()
}

#[tauri::command]
async fn readiness_action(rec: Rec<'_>, action: String) -> Result<(), String> {
    rec.inner().readiness_action(&action).await;
    Ok(())
}

#[tauri::command]
async fn check_update(app: tauri::AppHandle, force: bool) -> Result<(), String> {
    updater::check(&app, force).await;
    Ok(())
}

// --- kolejne etapy ---

#[tauri::command]
fn claude_note(rec: Rec) -> String {
    rec.markdown(None)
}

#[tauri::command]
fn import_file(_path: String) -> Result<(), String> {
    Err(NOT_YET.into())
}

#[tauri::command]
fn ask(_question: String, _image: Option<String>) -> Result<(), String> {
    Err(NOT_YET.into())
}

#[tauri::command]
fn clipboard_image() -> Option<serde_json::Value> {
    None
}

#[tauri::command]
fn toggle_overlay() -> Result<bool, String> {
    Err(NOT_YET.into())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        // Druga kopia aplikacji ma tylko pokazać pierwszą: dwa nasłuchy naraz
        // biłyby się o port whispera.
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            if let Some(window) = app.get_webview_window("main") {
                let _ = window.unminimize();
                let _ = window.show();
                let _ = window.set_focus();
            }
        }))
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_process::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_clipboard_manager::init())
        .setup(|app| {
            let vendor = app
                .path()
                .resource_dir()
                .map(|d| d.join("vendor"))
                .unwrap_or_else(|_| std::path::PathBuf::from("vendor"));
            let rec = Recorder::new(app.handle().clone(), vendor);
            app.manage(rec);
            updater::spawn(app.handle().clone());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_state,
            get_settings,
            save_settings,
            list_mics,
            models,
            start,
            stop,
            cancel_processing,
            chat_text,
            markdown,
            json_export,
            write_file,
            readiness,
            readiness_action,
            check_update,
            claude_note,
            import_file,
            ask,
            clipboard_image,
            toggle_overlay,
        ])
        .run(tauri::generate_context!())
        .expect("błąd uruchamiania call-whisper");
}
