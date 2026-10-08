//! Współpraca z OBS: pomyłka w haśle kończy się cichym „czekam na OBS",
//! a w zdarzeniach - transkryptem, który nigdy nie powstaje.
use std::path::Path;

use cw_core::*;
use serde_json::{json, Value};

#[test]
fn uwierzytelnienie_zgadza_sie_z_node() {
    // Wartość policzona niezależnie przez node:crypto, tym samym wzorem.
    let auth = Obs::authentication(
        "supersecret",
        "lM1GncleQOaCu9lT1yeUZhFYnqhsLLP1G5lAGo3ixaI=",
        "+IxH4CnCiqpX1rM9scsNynZzbOe4KhDeYcTNS3PDaeY=",
    );
    assert_eq!(auth, "sQBlPUYd9mki/3XVFBp4Pt08FCMWdMVIqnFWdEitUME=");
}

#[test]
fn konfiguracja_z_pliku_obs() {
    let j = r#"{"alerts_enabled":false,"auth_required":true,"first_load":false,"server_enabled":true,"server_password":"abc","server_port":4456}"#;
    assert_eq!(ServerConfig::parse(j.as_bytes()), Some(ServerConfig { enabled: true, port: 4456, password: Some("abc".into()) }));

    // Bez wymaganego hasła nie wysyłamy go wcale.
    let open = r#"{"auth_required":false,"server_enabled":false,"server_password":"abc","server_port":4455}"#;
    assert_eq!(ServerConfig::parse(open.as_bytes()), Some(ServerConfig { enabled: false, port: 4455, password: None }));
    assert_eq!(ServerConfig::parse(b"nie json"), None);
}

#[test]
fn wlaczenie_serwera_zostawia_reszte_pliku() {
    let unique = format!("{}-{}", std::process::id(), cw_core::now_ms() as u64);
    let path = std::env::temp_dir().join(format!("obs-{unique}.json"));
    std::fs::write(
        &path,
        r#"{"auth_required":true,"server_enabled":false,"server_password":"abc","server_port":4455,"alerts_enabled":false}"#,
    )
    .unwrap();

    assert!(ServerConfig::enable_server(&path));
    let j: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!(j["server_enabled"], true);
    assert_eq!(j["server_password"], "abc");
    assert_eq!(j["alerts_enabled"], false);

    // Brak pliku: nic nie zakładamy.
    let missing = std::env::temp_dir().join(format!("brak-{unique}.json"));
    assert!(!ServerConfig::enable_server(&missing));
    assert!(!missing.exists());
}

#[test]
fn transkrypt_obok_nagrania() {
    assert_eq!(
        Obs::transcript_path("/Users/x/Movies/2026-10-05 22-30-00.mkv"),
        Path::new("/Users/x/Movies/2026-10-05 22-30-00.md")
    );
}

#[test]
fn zdarzenia_nagrywania() {
    fn event(state: &str, path: Option<&str>) -> Value {
        let mut data = json!({ "outputActive": state.ends_with("STARTED"), "outputState": state });
        if let Some(p) = path {
            data["outputPath"] = json!(p);
        }
        json!({ "op": 5, "d": { "eventType": "RecordStateChanged", "eventIntent": 64, "eventData": data } })
    }
    assert_eq!(
        Obs::record_event(&event("OBS_WEBSOCKET_OUTPUT_STARTED", Some("/a.mkv"))),
        Some(RecordEvent::Started { path: Some("/a.mkv".into()) })
    );
    assert_eq!(
        Obs::record_event(&event("OBS_WEBSOCKET_OUTPUT_STOPPED", Some("/a.mkv"))),
        Some(RecordEvent::Stopped { path: Some("/a.mkv".into()) })
    );
    assert_eq!(Obs::record_event(&event("OBS_WEBSOCKET_OUTPUT_STOPPED", Some(""))), Some(RecordEvent::Stopped { path: None }));
    assert_eq!(Obs::record_event(&event("OBS_WEBSOCKET_OUTPUT_STOPPING", None)), None);
    assert_eq!(Obs::record_event(&event("OBS_WEBSOCKET_OUTPUT_PAUSED", None)), None);
    assert_eq!(Obs::record_event(&json!({ "op": 5, "d": { "eventType": "StreamStateChanged", "eventData": {} } })), None);
}
