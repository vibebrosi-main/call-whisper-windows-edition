//! Testy zgodności z implementacją webową.
//!
//! Wektory referencyjne pochodzą z działającego kodu JS
//! (`macos/tools/gen-fixtures.mjs`), a nie z ręcznie przepisanych asercji.
//! Dzięki temu każda rozbieżność portu wychodzi tu, a nie w trakcie rozmowy.
mod common;

use common::*;
use cw_core::*;

// MARK: - tekst

#[test]
fn reconcile_zgadza_sie_z_js() {
    let items = cases("reconcileCases");
    assert!(!items.is_empty());
    for c in items {
        let (prev, next) = (s(&c["prev"]), s(&c["next"]));
        assert_eq!(Text::reconcile(prev, next), s(&c["result"]), "reconcile({prev:?}, {next:?})");
    }
}

#[test]
fn normalize_zgadza_sie_z_js() {
    for c in cases("normalizeCases") {
        assert_eq!(Text::normalize(c["input"].as_str().unwrap_or("")), s(&c["result"]));
    }
}

#[test]
fn fold_zgadza_sie_z_js() {
    for c in cases("foldCases") {
        let input = s(&c["input"]);
        assert_eq!(Text::fold(input), s(&c["result"]), "fold({input:?})");
    }
}

#[test]
fn word_count_i_truncate() {
    for c in cases("wordCountCases") {
        assert_eq!(Text::word_count(s(&c["input"])) as u64, c["result"].as_u64().unwrap());
    }
    for c in cases("truncateCases") {
        assert_eq!(Text::truncate(s(&c["input"]), c["max"].as_u64().unwrap() as usize), s(&c["result"]));
    }
}

// MARK: - czas

#[test]
fn formatowanie_czasu() {
    for c in cases("offsetCases") {
        let ms = c["ms"].as_f64().unwrap();
        assert_eq!(TimeFormat::offset(ms), s(&c["result"]), "offset({ms})");
    }
    for c in cases("durationCases") {
        let ms = c["ms"].as_f64().unwrap();
        assert_eq!(TimeFormat::duration(ms), s(&c["result"]), "duration({ms})");
    }
}

// MARK: - wykrywanie pytań

#[test]
fn wykrywanie_pytan_zgadza_sie_z_js() {
    let items = cases("questionCases");
    assert!(items.len() >= 10);
    let mut failures = Vec::new();
    for c in items {
        let text = s(&c["text"]);
        let v = QuestionDetector::detect(text);
        let ok = v.is_question == c["isQuestion"].as_bool().unwrap()
            && (v.confidence - c["confidence"].as_f64().unwrap()).abs() < 1e-9
            && v.reason == s(&c["reason"])
            && v.question == s(&c["question"]);
        if !ok {
            failures.push(format!("{text:?}\n  got: {v:?}\n  exp: {c}"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn podzial_na_frazy() {
    for c in cases("clauseCases") {
        let expected: Vec<String> = c["result"].as_array().unwrap().iter().map(|v| s(v).to_string()).collect();
        assert_eq!(QuestionDetector::split_clauses(s(&c["text"])), expected);
    }
}

#[test]
fn prompt_zgadza_sie_z_js() {
    let segments = markdown_segments();
    let prompt = build_prompt("A jakie RPO i RTO to nam daje?", &segments, "Standup zespołu", "", DEFAULT_CONTEXT_SEGMENTS);
    assert_eq!(prompt.as_deref(), Some(s(fx("prompt"))));
}

// MARK: - markdown

/// 2026-08-23 08:15:03 UTC.
fn started_at() -> f64 {
    use chrono::TimeZone;
    chrono::Utc.with_ymd_and_hms(2026, 8, 23, 8, 15, 3).unwrap().timestamp_millis() as f64
}

fn markdown_segments() -> Vec<Segment> {
    let t = started_at();
    let seg = |id: &str, speaker: &str, text: &str, from: f64, to: f64, is_final: bool| Segment {
        id: id.into(),
        speaker: speaker.into(),
        text: text.into(),
        started_at: t + from,
        ended_at: t + to,
        offset_ms: from,
        is_final,
    };
    vec![
        seg("s1", "Anna Kowalska", "Cześć wszystkim, zaczynamy standup.", 4000.0, 9000.0, true),
        seg("s2", "Jan Nowak", "Hej, słychać mnie? Mam *gwiazdkę* i _podkreślenie_.", 11_000.0, 14_000.0, true),
        seg("s3", "Anna Kowalska", "Lecimy dalej.", 20_000.0, 22_000.0, false),
    ]
}

/// Fixture ma nagłówek w czasie lokalnym maszyny, na której powstał
/// (Europe/Warsaw: „2026-08-23 10:15"). Podstawiamy czas lokalny tej maszyny,
/// żeby test nie zależał od strefy.
fn localized(expected: &str) -> String {
    expected.replace("2026-08-23 10:15", &TimeFormat::local_date_time(started_at(), false))
}

#[test]
fn markdown_zgadza_sie_z_js() {
    let meta = SessionMeta {
        title: "Standup zespołu".into(),
        source: "google-meet".into(),
        url: "https://meet.google.com/abc-defg-hij".into(),
        started_at: Some(started_at()),
        ended_at: Some(started_at() + 2_531_000.0),
    };
    let session = Session { meta, segments: markdown_segments() };

    let opts = MarkdownOptions { frontmatter: false, ..Default::default() };
    assert_eq!(Markdown::render(&session, &opts), localized(s(fx("markdown"))));

    let opts_en = MarkdownOptions { frontmatter: false, locale: "en".into(), stats: false, ..Default::default() };
    assert_eq!(Markdown::render(&session, &opts_en), localized(s(fx("markdownEn"))));
}

// MARK: - transkrypt

#[test]
fn transcript_store_zgadza_sie_z_js() {
    let t0 = 1_700_000_000_000.0;
    let mut store = TranscriptStore::with_options(t0, 2500.0, 2500.0, 60_000.0, 1);
    store.upsert("a", Some("Anna"), "Cześć", t0 + 100.0, false);
    store.upsert("a", Some("Anna"), "Cześć wszystkim", t0 + 400.0, false);
    store.upsert("b", Some("Jan"), "Hej", t0 + 900.0, false);
    store.finalize_idle(t0 + 5000.0);
    store.upsert("c", Some("Anna"), "Druga wypowiedź Anny", t0 + 30_000.0, false);
    store.finalize_all(t0 + 40_000.0);

    let expected = cases("transcriptResult");
    let actual = store.segments();
    assert_eq!(actual.len(), expected.len(), "liczba segmentów");
    for (a, e) in actual.iter().zip(expected) {
        assert_eq!(a.speaker, s(&e["speaker"]));
        assert_eq!(a.text, s(&e["text"]));
        assert_eq!(a.offset_ms, e["offsetMs"].as_f64().unwrap());
        assert_eq!(a.is_final, e["final"].as_bool().unwrap());
    }
}

/// Segment przekracza granicę IPC jako JSON — pola jak w JS/Swift.
#[test]
fn segment_serializuje_sie_jak_w_js() {
    let seg = &markdown_segments()[2];
    let json = serde_json::to_value(seg).unwrap();
    for key in ["id", "speaker", "text", "startedAt", "endedAt", "offsetMs", "final"] {
        assert!(json.get(key).is_some(), "brak pola {key}: {json}");
    }
    assert_eq!(json["final"], false);
    let back: Segment = serde_json::from_value(json).unwrap();
    assert_eq!(&back, seg);
}
