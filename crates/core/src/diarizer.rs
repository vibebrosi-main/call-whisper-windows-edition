//! Rozpoznawanie mówcy po głosie (diaryzacja) — online, bez modelu neuronowego.
//!
//! Łańcuch: ramki PCM -> VAD -> MFCC -> embedding wypowiedzi -> klastrowanie
//! online po podobieństwie kosinusowym. Embedding to statystyki cepstralne
//! (średnia + odchylenie), czyli klasyczne podejście sprzed ery sieci
//! neuronowych.
//!
//! Świadome ograniczenie: to rozdziela wyraźnie różne głosy, ale jest istotnie
//! słabsze od modeli typu x-vector/ECAPA. Podobne głosy potrafi skleić.
//! Port z `Diarizer.swift` (a ten z `extension/src/adapters/audio/diarizer.js`).

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::dsp::{MfccExtractor, Vad};

/// Kosinus między wektorami znormalizowanymi L2.
pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Normalizacja L2.
pub fn l2_normalize(vector: &[f64]) -> Vec<f64> {
    let norm = vector.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm > 0.0 {
        vector.iter().map(|v| v / norm).collect()
    } else {
        vector.to_vec()
    }
}

/// Embedding wypowiedzi: [średnia MFCC, odchylenie MFCC], znormalizowany L2.
pub fn embed_frames(frames: &[Vec<f64>]) -> Option<Vec<f64>> {
    let dim = frames.first()?.len();
    let count = frames.len() as f64;
    let mut mean = vec![0.0; dim];
    let mut variance = vec![0.0; dim];

    for frame in frames {
        for i in 0..dim {
            mean[i] += frame[i];
        }
    }
    for m in &mut mean {
        *m /= count;
    }

    for frame in frames {
        for i in 0..dim {
            let d = frame[i] - mean[i];
            variance[i] += d * d;
        }
    }
    for v in &mut variance {
        *v = (*v / count).sqrt();
    }

    let mut embedding = mean;
    embedding.extend(variance);
    Some(l2_normalize(&embedding))
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Assignment {
    pub index: usize,
    pub similarity: f64,
    pub is_new: bool,
}

/// Klastrowanie online: przypisuje embedding do mówcy albo zakłada nowego.
#[derive(Debug, Clone)]
pub struct SpeakerTracker {
    /// Powyżej tego podobieństwa kosinusowego to ten sam mówca.
    pub threshold: f64,
    pub max_speakers: usize,
    /// Inercja centroidu — nowa próbka nie może go przewrócić.
    pub inertia: f64,
    centroids: Vec<Vec<f64>>,
    counts: Vec<usize>,
}

impl Default for SpeakerTracker {
    fn default() -> Self {
        Self::new(0.82, 8, 0.85)
    }
}

impl SpeakerTracker {
    pub fn new(threshold: f64, max_speakers: usize, centroid_inertia: f64) -> Self {
        Self { threshold, max_speakers, inertia: centroid_inertia, centroids: Vec::new(), counts: Vec::new() }
    }

    pub fn assign(&mut self, embedding: &[f64]) -> Assignment {
        let mut best: Option<usize> = None;
        let mut best_similarity = f64::NEG_INFINITY;

        for (i, c) in self.centroids.iter().enumerate() {
            let similarity = cosine_similarity(embedding, c);
            if similarity > best_similarity {
                best_similarity = similarity;
                best = Some(i);
            }
        }

        if let Some(b) = best {
            if best_similarity >= self.threshold {
                self.update(b, embedding);
                return Assignment { index: b, similarity: best_similarity, is_new: false };
            }
        }

        if self.centroids.len() < self.max_speakers {
            self.centroids.push(embedding.to_vec());
            self.counts.push(1);
            return Assignment { index: self.centroids.len() - 1, similarity: best_similarity, is_new: true };
        }

        // Limit mówców wyczerpany — dokładamy do najbliższego zamiast zgadywać.
        // (Swift przy max_speakers == 0 wywracał się na indeksie -1.)
        let b = best.expect("SpeakerTracker: max_speakers musi być > 0");
        self.update(b, embedding);
        Assignment { index: b, similarity: best_similarity, is_new: false }
    }

    fn update(&mut self, index: usize, embedding: &[f64]) {
        let inertia = self.inertia;
        let centroid: Vec<f64> =
            self.centroids[index].iter().zip(embedding).map(|(c, e)| c * inertia + e * (1.0 - inertia)).collect();
        self.centroids[index] = l2_normalize(&centroid);
        self.counts[index] += 1;
    }

    pub fn count(&self) -> usize {
        self.centroids.len()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Turn {
    pub speaker: usize,
    pub start_ms: f64,
    pub end_ms: f64,
}

/// Pełny diaryzator: karmisz go ramkami PCM, oddaje etykiety mówców i tury.
pub struct Diarizer {
    pub extractor: MfccExtractor,
    pub vad: Vad,
    pub tracker: SpeakerTracker,
    /// Ile ramek mowy musi się zebrać, zanim w ogóle zgadujemy mówcę.
    pub min_frames: usize,
    /// Co ile ramek odświeżamy prowizoryczną etykietę w trakcie mówienia.
    pub refresh_every_frames: usize,

    /// Czy w ogóle rozpoznawać, KTO mówi.
    ///
    /// Wyłączone zostawia wykrywanie mowy (VAD) i granice tur — na nich stoi
    /// cały łańcuch transkrypcji — a pomija MFCC, embedding i klastrowanie.
    /// Dwa powody, żeby móc to wyłączyć:
    ///
    ///  - klastrowanie MFCC to podejście sprzed ery sieci neuronowych i przy
    ///    kilku podobnych głosach rozsypuje jedną osobę na pięć etykiet, co
    ///    psuje transkrypt bardziej, niż brak etykiet w ogóle;
    ///  - MFCC liczone dla każdej ramki co 10 ms to najdroższa część pętli,
    ///    więc bez niego zostaje sam VAD na energii.
    pub identify_speakers: bool,

    /// Wołane przy zamknięciu tury — backend plikowy (whisper.cpp) tnie tu audio.
    pub on_turn: Box<dyn FnMut(Turn) + Send>,
    /// Wołane przy otwarciu tury — backend strumieniowy zaczyna tu nasłuch.
    pub on_turn_start: Box<dyn FnMut(f64) + Send>,

    current_speaker: Option<usize>,
    last_frame_loud: bool,
    turns: Vec<Turn>,

    frames: Vec<Vec<f64>>,
    turn_start_ms: f64,
    frames_since_refresh: usize,
    last_frame_ms: f64,
}

impl Default for Diarizer {
    fn default() -> Self {
        Self::new(MfccExtractor::default(), Vad::default(), SpeakerTracker::default(), 25, 25, true)
    }
}

impl Diarizer {
    pub fn new(
        extractor: MfccExtractor,
        vad: Vad,
        tracker: SpeakerTracker,
        min_frames: usize,
        refresh_every_frames: usize,
        identify_speakers: bool,
    ) -> Self {
        Self {
            extractor,
            vad,
            tracker,
            min_frames,
            refresh_every_frames,
            identify_speakers,
            on_turn: Box::new(|_| {}),
            on_turn_start: Box::new(|_| {}),
            current_speaker: None,
            last_frame_loud: false,
            turns: Vec::new(),
            frames: Vec::new(),
            turn_start_ms: 0.0,
            frames_since_refresh: 0,
            last_frame_ms: 0.0,
        }
    }

    /// Indeks mówcy aktualnie mówiącego, albo `None`.
    pub fn current_speaker(&self) -> Option<usize> {
        self.current_speaker
    }

    /// Stan VAD z ostatniej ramki — pozwala domknąć turę w przerwie między
    /// zdaniami, zamiast ciąć w połowie słowa.
    pub fn last_frame_loud(&self) -> bool {
        self.last_frame_loud
    }

    pub fn turns(&self) -> &[Turn] {
        &self.turns
    }

    /// Jedna ramka PCM (frame_size próbek), `t_ms` to czas jej początku.
    pub fn push_frame(&mut self, frame: &[f32], t_ms: f64) -> Option<usize> {
        let energy_db = MfccExtractor::frame_energy_db(frame);
        self.last_frame_ms = t_ms;
        let state = self.vad.push(energy_db);
        self.last_frame_loud = state.loud;

        if state.started {
            self.frames.clear();
            self.frames_since_refresh = 0;
            self.turn_start_ms = t_ms;
            // Bez rozpoznawania mówcy i tak trzeba mieć kogoś w turze, inaczej
            // `close_turn` uznałby ją za pustą i wyrzucił razem z transkrypcją.
            self.current_speaker = if self.identify_speakers { None } else { Some(0) };
            (self.on_turn_start)(t_ms);
        }

        if self.identify_speakers && state.speaking && state.loud {
            self.frames.push(self.extractor.frame_to_mfcc(frame));
            self.frames_since_refresh += 1;

            let enough = self.frames.len() >= self.min_frames;
            let due = self.frames_since_refresh >= self.refresh_every_frames;
            if enough && (self.current_speaker.is_none() || due) {
                self.frames_since_refresh = 0;
                self.classify();
            }
        }

        if state.ended {
            self.close_turn(t_ms);
        }
        self.current_speaker
    }

    fn classify(&mut self) {
        // Świadomie bez CMN: w obrębie jednego źródła kanał jest stały, więc
        // normalizacja cepstralna nic nie różnicuje, a jej dryf w trakcie sesji
        // przesuwa przestrzeń embeddingów pod już nauczonymi centroidami.
        if let Some(embedding) = embed_frames(&self.frames) {
            self.current_speaker = Some(self.tracker.assign(&embedding).index);
        }
    }

    fn close_turn(&mut self, end_ms: f64) {
        if let Some(speaker) = self.current_speaker {
            if end_ms > self.turn_start_ms {
                let turn = Turn { speaker, start_ms: self.turn_start_ms, end_ms };
                self.turns.push(turn);
                (self.on_turn)(turn);
            }
        }
        self.frames.clear();
        self.current_speaker = None;
    }

    /// Domyka otwartą turę — na koniec sesji.
    pub fn flush(&mut self, end_ms: f64) {
        if self.vad.speaking() {
            if self.current_speaker.is_none() && !self.frames.is_empty() {
                self.classify();
            }
            self.close_turn(end_ms);
            self.vad.reset();
        }
    }

    /// Kto dominował w oknie czasu — tak wiążemy tekst z ASR z mówcą.
    /// Uwzględnia też turę wciąż otwartą.
    pub fn dominant_speaker(&self, from_ms: f64, to_ms: f64) -> Option<usize> {
        let mut totals: HashMap<usize, f64> = HashMap::new();
        let mut add = |speaker: usize, ms: f64| {
            if ms > 0.0 {
                *totals.entry(speaker).or_insert(0.0) += ms;
            }
        };

        for turn in &self.turns {
            add(turn.speaker, turn.end_ms.min(to_ms) - turn.start_ms.max(from_ms));
        }
        if let Some(speaker) = self.current_speaker {
            // Tura wciąż otwarta — jej koniec to ostatnia widziana ramka, nie zegar ścienny.
            add(speaker, to_ms.min(self.last_frame_ms) - self.turn_start_ms.max(from_ms));
        }

        // Największy udział; remis wygrywa niższy indeks.
        let mut best: Option<(usize, f64)> = None;
        for (&k, &v) in &totals {
            match best {
                Some((bk, bv)) if !(v > bv || (v == bv && k < bk)) => {}
                _ => best = Some((k, v)),
            }
        }
        best.map(|(k, _)| k)
    }

    pub fn speaker_count(&self) -> usize {
        self.tracker.count()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Frame {
    pub frame: Vec<f32>,
    pub start_sample: usize,
}

/// Tnie ciągły strumień próbek na ramki o stałej długości z zadanym skokiem.
/// Port z `extension/src/adapters/audio/framer.js`.
#[derive(Debug, Clone)]
pub struct Framer {
    pub frame_size: usize,
    pub hop_size: usize,
    buffer: Vec<f32>,
    /// Ile próbek już wypadło z bufora — pozwala liczyć czas ramki bez dryfu.
    consumed: usize,
}

impl Framer {
    pub fn new(frame_size: usize, hop_size: usize) -> Self {
        assert!(frame_size > 0 && hop_size > 0, "Framer: rozmiary muszą być dodatnie");
        Self { frame_size, hop_size, buffer: Vec::new(), consumed: 0 }
    }

    /// Dokłada próbki i oddaje wszystkie kompletne ramki wraz z indeksem
    /// pierwszej próbki każdej z nich.
    pub fn push(&mut self, samples: &[f32]) -> Vec<Frame> {
        self.buffer.extend_from_slice(samples);
        let mut out = Vec::new();
        while self.buffer.len() >= self.frame_size {
            out.push(Frame { frame: self.buffer[..self.frame_size].to_vec(), start_sample: self.consumed });
            // Swift `removeFirst(hopSize)` wywraca się, gdy skok > bufora;
            // tu ucinamy do długości bufora.
            let n = self.hop_size.min(self.buffer.len());
            self.buffer.drain(..n);
            self.consumed += self.hop_size;
        }
        out
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
        self.consumed = 0;
    }
}
