//! Logika przeniesiona z OpenWhispr/whistler: diaryzacja neuronowa,
//! wykrywanie rozmów, import nagrań.
#![allow(clippy::approx_constant)]
use std::collections::HashSet;

use cw_core::*;

// Prawdziwe wyjście sherpa-onnx, łącznie z liniami, których nie chcemy.
const OUTPUT: &str = "Started
0.318 -- 6.865 speaker_00
7.017 -- 10.747 speaker_01
11.455 -- 13.632 speaker_01
Duration : 56.861 s
Elapsed seconds: 2.979 s";

fn turn(start: f64, end: f64, cluster: &str) -> SpeakerTurn {
    SpeakerTurn { start, end, cluster: cluster.into() }
}

fn tt(start: f64, end: f64, text: &str) -> TimedText {
    TimedText { start, end, text: text.into() }
}

fn strings(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

#[test]
fn parsuje_tylko_tury() {
    let turns = SpeakerTurns::parse(OUTPUT);
    assert_eq!(turns.len(), 3);
    assert_eq!(turns[0], turn(0.318, 6.865, "speaker_00"));
    assert_eq!(turns[2].cluster, "speaker_01");
}

#[test]
fn wygrywa_najdluzsze_pokrycie() {
    let turns = SpeakerTurns::parse(OUTPUT);
    // 6,0-8,0: speaker_00 pokrywa 0,865 s, speaker_01 0,983 s.
    assert_eq!(SpeakerTurns::dominant_cluster(6.0, 8.0, &turns).as_deref(), Some("speaker_01"));
    assert_eq!(SpeakerTurns::dominant_cluster(1.0, 2.0, &turns).as_deref(), Some("speaker_00"));
    assert_eq!(SpeakerTurns::dominant_cluster(30.0, 31.0, &turns), None);
}

#[test]
fn numeracja_w_kolejnosci_odezwania() {
    // Klaster o wyższym numerze odzywa się pierwszy — ma dostać „1".
    let turns = [turn(0.0, 5.0, "speaker_07"), turn(5.0, 9.0, "speaker_02"), turn(9.0, 12.0, "speaker_07")];
    let texts = [
        tt(0.5, 4.0, "a"),
        tt(5.5, 8.0, "b"),
        tt(9.5, 11.0, "c"),
        // Poza turami: dostaje najbliższy klaster, nie etykietę zapasową.
        tt(13.0, 14.0, "d"),
    ];
    let labels = SpeakerTurns::label(&texts, &turns, "Osoba", "Nagranie");
    assert_eq!(labels, strings(&["Osoba 1", "Osoba 2", "Osoba 1", "Osoba 1"]));
}

#[test]
fn bez_tur_etykieta_zapasowa() {
    let texts = [tt(0.0, 1.0, "a")];
    assert_eq!(SpeakerTurns::label(&texts, &[], "Osoba", "Nagranie"), strings(&["Nagranie"]));
}

#[test]
fn skleja_akapity_tej_samej_osoby() {
    let texts = [tt(0.0, 2.0, " Cześć."), tt(2.5, 4.0, "Zaczynamy."), tt(4.2, 6.0, "Hej."), tt(20.0, 22.0, "Po przerwie.")];
    let speakers = strings(&["Osoba 1", "Osoba 1", "Osoba 2", "Osoba 2"]);
    let paragraphs = SpeakerTurns::paragraphs(&texts, &speakers, 2.5, 60.0);
    assert_eq!(paragraphs.len(), 3);
    assert_eq!(paragraphs[0].item.text, "Cześć. Zaczynamy.");
    assert_eq!(paragraphs[0].item.end, 4.0);
    assert_eq!(paragraphs[2].item.text, "Po przerwie.");

    let segments = SpeakerTurns::segments(&paragraphs, 1_000_000.0, "imp");
    assert_eq!(segments[1].offset_ms, 4200.0);
    assert_eq!(segments[1].started_at, 1_004_200.0);
    assert!(segments.iter().all(|s| s.is_final));
}

#[test]
fn relabel_nie_rusza_mikrofonu() {
    let base = 1_000_000.0;
    let seg = |id: &str, speaker: &str, at: f64, len: f64| Segment {
        id: id.into(),
        speaker: speaker.into(),
        text: id.into(),
        started_at: base + at * 1000.0,
        ended_at: base + (at + len) * 1000.0,
        offset_ms: at * 1000.0,
        is_final: true,
    };
    let segments = [seg("a", "Rozmówcy", 0.0, 3.0), seg("b", "Ty", 3.0, 2.0), seg("c", "Rozmówcy", 6.0, 3.0), seg("d", "Rozmówcy", 10.0, 2.0)];
    let turns = [turn(0.0, 3.0, "speaker_00"), turn(6.0, 9.0, "speaker_01"), turn(10.0, 12.0, "speaker_00")];
    let out = SpeakerTurns::relabel(&segments, &turns, SpeakerTurns::is_system_label, "Rozmówca");
    let names: Vec<&str> = out.iter().map(|s| s.speaker.as_str()).collect();
    assert_eq!(names, ["Rozmówca 1", "Ty", "Rozmówca 2", "Rozmówca 1"]);
}

#[test]
fn relabel_jednego_glosu_zostawia_etykiete() {
    let segments = [Segment {
        id: "a".into(),
        speaker: "Rozmówcy".into(),
        text: "a".into(),
        started_at: 0.0,
        ended_at: 2000.0,
        offset_ms: 0.0,
        is_final: true,
    }];
    let turns = [turn(0.0, 2.0, "speaker_00")];
    assert_eq!(SpeakerTurns::relabel(&segments, &turns, SpeakerTurns::is_system_label, "Rozmówca")[0].speaker, "Rozmówcy");
}

fn set(v: &[&str]) -> HashSet<String> {
    v.iter().map(|s| s.to_string()).collect()
}

#[test]
fn potrzeba_mikrofonu_i_aplikacji() {
    assert_eq!(MeetingDetection::active_meeting(false, &set(&["us.zoom.xos"]), None), None);
    assert_eq!(MeetingDetection::active_meeting(true, &set(&[]), None), None);
    assert_eq!(MeetingDetection::active_meeting(true, &set(&["us.zoom.xos"]), None), Some("Zoom"));
}

#[test]
fn pierwszy_plan_wygrywa() {
    let running = set(&["com.tinyspeck.slackmacgap", "com.microsoft.teams2"]);
    assert_eq!(MeetingDetection::active_meeting(true, &running, Some("com.microsoft.teams2")), Some("Microsoft Teams"));
}

#[test]
fn przegladarka_tylko_na_pierwszym_planie() {
    assert_eq!(
        MeetingDetection::active_meeting(true, &set(&["com.google.Chrome"]), Some("com.google.Chrome")),
        Some("przeglądarce")
    );
    assert_eq!(MeetingDetection::active_meeting(true, &set(&["com.google.Chrome"]), Some("com.apple.finder")), None);
}

#[test]
fn histereza() {
    let mut d = Debouncer::new(2, 3);
    assert_eq!(d.feed(Some("Zoom")), None);
    assert_eq!(d.feed(None), None); // mrugnięcie zeruje licznik
    assert_eq!(d.feed(Some("Zoom")), None);
    assert_eq!(d.feed(Some("Zoom")), Some(MeetingChange::Started("Zoom".into())));
    assert_eq!(d.feed(Some("Zoom")), None);
    assert_eq!(d.feed(None), None);
    assert_eq!(d.feed(None), None);
    assert_eq!(d.feed(None), Some(MeetingChange::Ended("Zoom".into())));
    assert_eq!(d.active(), None);
}
