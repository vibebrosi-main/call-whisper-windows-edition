//! Tekst do wklejenia w czat: jedna linia na wypowiedź, bez pustych.
use cw_core::*;

fn seg(id: &str, speaker: &str, text: &str, offset_ms: f64, is_final: bool) -> Segment {
    Segment {
        id: id.into(),
        speaker: speaker.into(),
        text: text.into(),
        started_at: 0.0,
        ended_at: 0.0,
        offset_ms,
        is_final,
    }
}

#[test]
fn jedna_linia_na_wypowiedz() {
    let segments = [
        seg("a", "Rozmówcy", "Ile to trwało?", 484_000.0, true),
        seg("b", "Ty", "  ", 490_000.0, true),
        seg("c", "Ty", "Dwa  lata\nna etacie.", 3_661_000.0, false),
    ];
    assert_eq!(Markdown::chat_text(&segments), "[00:08:04] Rozmówcy: Ile to trwało?\n[01:01:01] Ty: Dwa lata na etacie.");
    assert_eq!(Markdown::chat_text(&[]), "");
}
