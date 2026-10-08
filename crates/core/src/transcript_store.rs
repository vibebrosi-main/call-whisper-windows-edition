//! TranscriptStore — rdzeń niezależny od źródła.
//!
//! Adapter (przechwytywanie dźwięku, mikrofon, whisper.cpp…) woła `upsert()`
//! z migawką aktualnie rozpoznanego bloku. Store zajmuje się resztą: scalaniem
//! strumienia, pilnowaniem tożsamości mówcy, finalizacją po ciszy i łączeniem
//! poszatkowanych wypowiedzi tej samej osoby.
//!
//! Port z `TranscriptStore.swift` (a ten z `extension/src/core/transcript.js`).

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};

use crate::text::{char_len, Text};
use crate::time::now_ms;

pub const UNKNOWN_SPEAKER: &str = "Nieznany";

/// Jedna wypowiedź w transkrypcie.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Segment {
    pub id: String,
    pub speaker: String,
    pub text: String,
    pub started_at: f64,
    pub ended_at: f64,
    pub offset_ms: f64,
    #[serde(rename = "final")]
    pub is_final: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeakerStats {
    pub name: String,
    pub segments: usize,
    pub chars: usize,
    pub words: usize,
    pub first_at: f64,
    pub talk_ms: f64,
}

#[derive(Debug, Clone)]
struct Live {
    id: String,
    key: Option<String>,
    speaker: String,
    text: String,
    started_at: f64,
    updated_at: f64,
    ended_at: Option<f64>,
    is_final: bool,
}

const RETIRED_LIMIT: usize = 100;
const SEALED_LIMIT: usize = 200;

#[derive(Debug)]
pub struct TranscriptStore {
    /// Po tylu ms bez zmiany tekstu segment uznajemy za domknięty.
    pub silence_ms: f64,
    /// Kolejny segment tej samej osoby w tym oknie doklejamy do poprzedniego.
    pub merge_gap_ms: f64,
    /// Nie sklejamy w nieskończoność — twardy limit długości akapitu.
    pub max_merged_ms: f64,
    /// Krótsze wypowiedzi niż tyle znaków są ignorowane przy finalizacji.
    pub min_chars: usize,

    started_at: f64,
    /// Rośnie przy każdej zmianie — tanie źródło prawdy dla UI.
    revision: u64,

    segment_list: Vec<Live>,
    /// Klucz bloku -> id segmentu na żywo (segment siedzi w `segment_list`).
    live: HashMap<String, String>,
    /// Treść ostatnio domkniętego segmentu per klucz — broni przed dublowaniem
    /// bloku, który źródło pokazuje jeszcze długo po końcu wypowiedzi.
    retired: HashMap<String, (String, String)>,
    retired_order: Vec<String>,
    /// Klucze domknięte ostatecznie — kolejne migawki są ignorowane.
    sealed: HashSet<String>,
    sealed_order: Vec<String>,
    id_counter: u64,
}

impl Default for TranscriptStore {
    fn default() -> Self {
        Self::new(now_ms())
    }
}

impl TranscriptStore {
    /// Domyślne progi: cisza 2500 ms, sklejanie 2500 ms, akapit do 60 s, min. 1 znak.
    pub fn new(started_at: f64) -> Self {
        Self::with_options(started_at, 2500.0, 2500.0, 60_000.0, 1)
    }

    pub fn with_options(started_at: f64, silence_ms: f64, merge_gap_ms: f64, max_merged_ms: f64, min_chars: usize) -> Self {
        Self {
            silence_ms,
            merge_gap_ms,
            max_merged_ms,
            min_chars,
            started_at,
            revision: 0,
            segment_list: Vec::new(),
            live: HashMap::new(),
            retired: HashMap::new(),
            retired_order: Vec::new(),
            sealed: HashSet::new(),
            sealed_order: Vec::new(),
            id_counter: 0,
        }
    }

    pub fn started_at(&self) -> f64 {
        self.started_at
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    fn index_of(&self, id: &str) -> Option<usize> {
        self.segment_list.iter().position(|s| s.id == id)
    }

    fn live_index(&self, key: &str) -> Option<usize> {
        self.live.get(key).and_then(|id| self.index_of(id))
    }

    /// Migawka aktualnego bloku rozpoznania.
    ///
    /// `replace: true` oznacza, że źródło podaje pełną, poprawioną treść przy
    /// każdej aktualizacji (tak działa SpeechTranscriber) — wtedy tekst
    /// podmieniamy zamiast scalać.
    pub fn upsert(&mut self, key: &str, speaker: Option<&str>, text: &str, at: f64, replace: bool) -> Option<Segment> {
        let body = Text::normalize(text);
        if body.is_empty() || self.sealed.contains(key) {
            return None;
        }
        let who_raw = Text::normalize(speaker.unwrap_or(""));
        let who = if who_raw.is_empty() { UNKNOWN_SPEAKER.to_string() } else { who_raw };

        let mut seg = self.live_index(key);

        // Ten sam blok przejęty przez innego mówcę => zamykamy poprzedni.
        if let Some(i) = seg {
            if self.segment_list[i].speaker != who {
                let id = self.segment_list[i].id.clone();
                self.close(&id, at);
                self.live.remove(key);
                seg = None;
            }
        }

        let mut incoming = body.clone();
        if seg.is_none() {
            if let Some((old_speaker, old_text)) = self.retired.get(key).cloned() {
                if old_speaker == who {
                    // Przy pełnych migawkach domknięte znaczy domknięte.
                    if replace {
                        return None;
                    }
                    let merged = Text::reconcile(&old_text, &body);
                    if merged == old_text {
                        return None; // blok wisi, nic nowego nie padło
                    }
                    incoming = if merged.starts_with(&old_text) {
                        Text::normalize(&merged.chars().skip(char_len(&old_text)).collect::<String>())
                    } else {
                        merged
                    };
                    self.drop_retired(key);
                    if incoming.is_empty() {
                        return None;
                    }
                }
            }
        }

        let Some(i) = seg else {
            self.id_counter += 1;
            let fresh = Live {
                id: format!("s{}", self.id_counter),
                key: Some(key.to_string()),
                speaker: who,
                text: incoming,
                started_at: at,
                updated_at: at,
                ended_at: None,
                is_final: false,
            };
            self.live.insert(key.to_string(), fresh.id.clone());
            let snap = self.snapshot(&fresh);
            self.segment_list.push(fresh);
            self.revision += 1;
            return Some(snap);
        };

        let current = self.segment_list[i].text.clone();
        let merged = if replace { body } else { Text::reconcile(&current, &body) };
        if merged != current {
            let s = &mut self.segment_list[i];
            s.text = merged;
            s.updated_at = at; // rośnie tylko przy realnej zmianie -> działa detekcja ciszy
            self.revision += 1;
        }
        Some(self.snapshot(&self.segment_list[i]))
    }

    /// Klucze na żywo w kolejności transkryptu. Swift iterował po słowniku
    /// (kolejność nieokreślona); tu ustalamy ją, żeby sklejanie było powtarzalne.
    fn live_keys_in_order(&self) -> Vec<(String, String)> {
        let mut keys: Vec<(usize, String, String)> = self
            .live
            .iter()
            .filter_map(|(k, id)| self.index_of(id).map(|i| (i, k.clone(), id.clone())))
            .collect();
        keys.sort_by_key(|(i, ..)| *i);
        keys.into_iter().map(|(_, k, id)| (k, id)).collect()
    }

    /// Domyka segmenty, które od `silence_ms` nic nie zmieniły.
    pub fn finalize_idle(&mut self, now: f64) {
        for (key, id) in self.live_keys_in_order() {
            let Some(i) = self.index_of(&id) else { continue };
            if now - self.segment_list[i].updated_at >= self.silence_ms {
                self.close(&id, now);
                self.live.remove(&key);
            }
        }
    }

    /// Blok zniknął ze źródła — wypowiedź na pewno się skończyła.
    pub fn drop_key(&mut self, key: &str, now: f64) {
        if let Some(id) = self.live.get(key).cloned() {
            self.close(&id, now);
            self.live.remove(key);
        }
        self.drop_retired(key);
    }

    /// Domyka segment i zamyka klucz na dobre — źródło może go wysłać ponownie.
    pub fn seal(&mut self, key: &str, now: f64) {
        if let Some(id) = self.live.get(key).cloned() {
            self.close(&id, now);
            self.live.remove(key);
        }
        self.drop_retired(key);
        if self.sealed.insert(key.to_string()) {
            self.sealed_order.push(key.to_string());
            if self.sealed_order.len() > SEALED_LIMIT {
                let oldest = self.sealed_order.remove(0);
                self.sealed.remove(&oldest);
            }
        }
    }

    /// Usuwa segment bez śladu — np. znacznik „w toku" po nieudanej transkrypcji.
    pub fn discard(&mut self, key: &str) -> bool {
        let Some(id) = self.live.get(key).cloned() else { return false };
        if let Some(i) = self.index_of(&id) {
            self.segment_list.remove(i);
        }
        self.live.remove(key);
        self.drop_retired(key);
        self.revision += 1;
        true
    }

    /// Koniec sesji.
    pub fn finalize_all(&mut self, now: f64) {
        for (_, id) in self.live_keys_in_order() {
            self.close(&id, now);
        }
        self.live.clear();
    }

    fn drop_retired(&mut self, key: &str) {
        if self.retired.remove(key).is_some() {
            self.retired_order.retain(|k| k != key);
        }
    }

    fn retire(&mut self, key: Option<String>, speaker: String, text: String) {
        let Some(key) = key else { return };
        if !self.retired.contains_key(&key) {
            self.retired_order.push(key.clone());
        }
        self.retired.insert(key, (speaker, text));
        if self.retired_order.len() > RETIRED_LIMIT {
            let oldest = self.retired_order.remove(0);
            self.retired.remove(&oldest);
        }
    }

    fn close(&mut self, id: &str, _now: f64) {
        let Some(i) = self.index_of(id) else { return };
        {
            let seg = &mut self.segment_list[i];
            seg.is_final = true;
            seg.ended_at = Some(seg.updated_at.max(seg.started_at));
        }
        let (key, speaker, text) = {
            let s = &self.segment_list[i];
            (s.key.clone(), s.speaker.clone(), s.text.clone())
        };
        self.retire(key, speaker, text);

        if char_len(&Text::normalize(&self.segment_list[i].text)) < self.min_chars {
            self.segment_list.remove(i);
            self.revision += 1;
            return;
        }

        // Sklejanie poszatkowanych wypowiedzi tej samej osoby.
        if i == 0 {
            self.revision += 1;
            return;
        }
        let seg = self.segment_list[i].clone();
        let prev = &mut self.segment_list[i - 1];
        if prev.is_final
            && prev.speaker == seg.speaker
            && seg.started_at - prev.ended_at.unwrap_or(prev.updated_at) <= self.merge_gap_ms
            && seg.updated_at - prev.started_at <= self.max_merged_ms
        {
            prev.text = Text::reconcile(&prev.text, &seg.text);
            prev.updated_at = seg.updated_at;
            prev.ended_at = seg.ended_at;
            self.segment_list.remove(i);
        }
        self.revision += 1;
    }

    fn snapshot(&self, s: &Live) -> Segment {
        Segment {
            id: s.id.clone(),
            speaker: s.speaker.clone(),
            text: s.text.clone(),
            started_at: s.started_at,
            ended_at: s.ended_at.unwrap_or(s.updated_at),
            offset_ms: (s.started_at - self.started_at).max(0.0),
            is_final: s.is_final,
        }
    }

    /// Wszystkie segmenty (domknięte i na żywo) z policzonym offsetem.
    pub fn segments(&self) -> Vec<Segment> {
        self.segment_list.iter().map(|s| self.snapshot(s)).collect()
    }

    pub fn live_count(&self) -> usize {
        self.live.len()
    }

    pub fn is_empty(&self) -> bool {
        self.segment_list.is_empty()
    }

    /// Statystyki per osoba, w kolejności pierwszego wystąpienia.
    pub fn speakers(&self) -> Vec<SpeakerStats> {
        let mut order: Vec<String> = Vec::new();
        let mut map: HashMap<String, SpeakerStats> = HashMap::new();
        for s in &self.segment_list {
            let entry = map.entry(s.speaker.clone()).or_insert_with(|| {
                order.push(s.speaker.clone());
                SpeakerStats { name: s.speaker.clone(), segments: 0, chars: 0, words: 0, first_at: s.started_at, talk_ms: 0.0 }
            });
            entry.segments += 1;
            entry.chars += char_len(&s.text);
            entry.words += Text::word_count(&s.text);
            entry.talk_ms += (s.ended_at.unwrap_or(s.updated_at) - s.started_at).max(0.0);
        }
        order.into_iter().filter_map(|n| map.remove(&n)).collect()
    }
}
