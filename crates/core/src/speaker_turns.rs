//! Łączenie wyniku diaryzacji z transkrypcją.
//!
//! Diaryzacja (pyannote + CAM++ przez sherpa-onnx, podejście z OpenWhispr)
//! i whisper liczą granice niezależnie, więc granice tur i segmentów tekstu
//! prawie nigdy się nie pokrywają. Etykietę dostaje ten klaster, który
//! pokrywa fragment tekstu najdłużej — nie ten, który był pierwszy.
//!
//! Port z `SpeakerTurns.swift`.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::transcript_store::{Segment, UNKNOWN_SPEAKER};

/// Tura mówcy z diaryzacji: przedział w sekundach i surowa etykieta klastra.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SpeakerTurn {
    pub start: f64,
    pub end: f64,
    pub cluster: String,
}

/// Fragment tekstu z osią czasu w sekundach (od początku nagrania).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TimedText {
    pub start: f64,
    pub end: f64,
    pub text: String,
}

/// Akapit jednej osoby (Swift: krotka `(speaker:, item:)`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Paragraph {
    pub speaker: String,
    pub item: TimedText,
}

pub struct SpeakerTurns;

fn distance(t: f64, turn: &SpeakerTurn) -> f64 {
    if t < turn.start {
        turn.start - t
    } else if t > turn.end {
        t - turn.end
    } else {
        0.0
    }
}

impl SpeakerTurns {
    /// Parsuje wyjście `sherpa-onnx-offline-speaker-diarization`:
    /// `0.031 -- 3.035 speaker_01`, jedna tura na linię. Resztę (logi, nagłówki)
    /// pomija.
    pub fn parse(output: &str) -> Vec<SpeakerTurn> {
        let mut turns = Vec::new();
        for raw in output.split(['\n', '\r', '\u{0B}', '\u{0C}', '\u{85}', '\u{2028}', '\u{2029}']) {
            let parts: Vec<&str> = raw.split(' ').filter(|p| !p.is_empty()).collect();
            if parts.len() != 4 || parts[1] != "--" || !parts[3].starts_with("speaker_") {
                continue;
            }
            let (Ok(start), Ok(end)) = (parts[0].parse::<f64>(), parts[2].parse::<f64>()) else { continue };
            // `end > start` odrzuca też NaN.
            if end.partial_cmp(&start) != Some(std::cmp::Ordering::Greater) {
                continue;
            }
            turns.push(SpeakerTurn { start, end, cluster: parts[3].to_string() });
        }
        turns.sort_by(|a, b| a.start.partial_cmp(&b.start).unwrap_or(std::cmp::Ordering::Equal));
        turns
    }

    /// Klaster, który najdłużej pokrywa przedział. `None`, gdy żaden.
    pub fn dominant_cluster(start: f64, end: f64, turns: &[SpeakerTurn]) -> Option<String> {
        let mut overlap: HashMap<&str, f64> = HashMap::new();
        for turn in turns.iter().filter(|t| t.end > start && t.start < end) {
            *overlap.entry(&turn.cluster).or_insert(0.0) += end.min(turn.end) - start.max(turn.start);
        }
        // Remis rozstrzygamy nazwą klastra, żeby wynik był deterministyczny.
        let mut best: Option<(&str, f64)> = None;
        for (&k, &v) in &overlap {
            match best {
                Some((bk, bv)) if !(v > bv || (v == bv && k < bk)) => {}
                _ => best = Some((k, v)),
            }
        }
        best.map(|(k, _)| k.to_string())
    }

    /// Najbliższy klaster dla fragmentu, którego nie pokrywa żadna tura —
    /// diaryzacja gubi krótkie wtrącenia, a tekst z nich jest prawdziwy.
    pub(crate) fn nearest_cluster(start: f64, end: f64, turns: &[SpeakerTurn]) -> Option<String> {
        let mid = (start + end) / 2.0;
        let mut best: Option<&SpeakerTurn> = None;
        for t in turns {
            if best.is_none_or(|b| distance(mid, t) < distance(mid, b)) {
                best = Some(t);
            }
        }
        best.map(|t| t.cluster.clone())
    }

    /// Przypisuje etykiety fragmentom tekstu.
    ///
    /// Surowe `speaker_07` z klastrowania nic nie mówią, więc numerujemy
    /// w kolejności pierwszego odezwania się: pierwszy głos w nagraniu to
    /// „{prefix} 1". Bez tur (diaryzacja wyłączona albo nieudana) wszystko
    /// dostaje `fallback`.
    pub fn label(texts: &[TimedText], turns: &[SpeakerTurn], prefix: &str, fallback: &str) -> Vec<String> {
        if turns.is_empty() {
            return texts.iter().map(|_| fallback.to_string()).collect();
        }
        let mut numbering: HashMap<String, usize> = HashMap::new();
        texts
            .iter()
            .map(|item| {
                let Some(cluster) = Self::dominant_cluster(item.start, item.end, turns)
                    .or_else(|| Self::nearest_cluster(item.start, item.end, turns))
                else {
                    return fallback.to_string();
                };
                let next = numbering.len() + 1;
                let n = *numbering.entry(cluster).or_insert(next);
                format!("{prefix} {n}")
            })
            .collect()
    }

    /// Skleja kolejne fragmenty tej samej osoby w akapity.
    ///
    /// Whisper tnie na zdania co kilka sekund; w notatce chcemy wypowiedzi,
    /// a nie osobny nagłówek nad każdym zdaniem. Sklejamy tylko przy krótkiej
    /// przerwie i do twardego limitu długości — tak samo jak `TranscriptStore`
    /// w trybie na żywo. Swift: domyślnie `max_gap = 2.5`, `max_length = 60`.
    pub fn paragraphs(texts: &[TimedText], speakers: &[String], max_gap: f64, max_length: f64) -> Vec<Paragraph> {
        let mut out: Vec<Paragraph> = Vec::new();
        for (item, speaker) in texts.iter().zip(speakers) {
            let text = item.text.trim();
            if text.is_empty() {
                continue;
            }
            if let Some(last) = out.last_mut() {
                if &last.speaker == speaker && item.start - last.item.end <= max_gap && item.end - last.item.start <= max_length {
                    last.item.text.push(' ');
                    last.item.text.push_str(text);
                    last.item.end = item.end;
                    continue;
                }
            }
            out.push(Paragraph {
                speaker: speaker.clone(),
                item: TimedText { start: item.start, end: item.end, text: text.to_string() },
            });
        }
        out
    }

    /// Gotowe segmenty transkryptu. `started_at` to czas ściany początku nagrania
    /// w ms — Markdown liczy z niego offsety i nagłówek. Swift: domyślnie `id_prefix = "imp"`.
    pub fn segments(paragraphs: &[Paragraph], started_at: f64, id_prefix: &str) -> Vec<Segment> {
        paragraphs
            .iter()
            .enumerate()
            .map(|(index, p)| Segment {
                id: format!("{id_prefix}-{index}"),
                speaker: p.speaker.clone(),
                text: p.item.text.clone(),
                started_at: started_at + p.item.start * 1000.0,
                ended_at: started_at + p.item.end * 1000.0,
                offset_ms: p.item.start * 1000.0,
                is_final: true,
            })
            .collect()
    }

    /// Przepisuje etykiety istniejących segmentów rozmowy na żywo.
    ///
    /// Dotyka tylko segmentów, dla których `is_target` zwraca prawdę (etykiety
    /// dźwięku systemu) — „Ty" z mikrofonu jest pewne, bo bierze się
    /// z rozdzielenia źródeł, a nie z barwy głosu, i diaryzacja nie ma go
    /// prawa nadpisać.
    pub fn relabel(segments: &[Segment], turns: &[SpeakerTurn], is_target: impl Fn(&str) -> bool, prefix: &str) -> Vec<Segment> {
        if turns.is_empty() {
            return segments.to_vec();
        }
        let targets: Vec<usize> = (0..segments.len()).filter(|&i| is_target(&segments[i].speaker)).collect();
        let texts: Vec<TimedText> = targets
            .iter()
            .map(|&i| {
                let s = &segments[i];
                let start = s.offset_ms / 1000.0;
                TimedText { start, end: (start + 0.1).max(start + (s.ended_at - s.started_at) / 1000.0), text: s.text.clone() }
            })
            .collect();
        let labels = Self::label(&texts, turns, prefix, prefix);
        // Jeden głos to żadna informacja — zostawiamy znaną etykietę.
        if labels.iter().collect::<HashSet<_>>().len() <= 1 {
            return segments.to_vec();
        }
        let mut out = segments.to_vec();
        for (&index, name) in targets.iter().zip(labels) {
            out[index].speaker = name;
        }
        out
    }

    /// Etykiety, które może nadać dźwięk systemu — w obu trybach (z MFCC
    /// i bez). To je przepisuje diaryzacja po rozmowie.
    pub fn is_system_label(speaker: &str) -> bool {
        speaker == "Rozmówcy" || speaker == UNKNOWN_SPEAKER || speaker.starts_with("Rozmówca ")
    }
}
