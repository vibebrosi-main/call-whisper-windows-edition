//! Słownictwo dla whispera wyciągane z kontekstu projektu.
//!
//! Ręczne słownictwo z Ustawień pokrywa stałe nazwy (React, Next.js), ale
//! nie to, co jest specyficzne dla danej rozmowy. Zmierzone 2026-10-06 na
//! `small`: z terminami z pliku kontekstu whisper pisał „Playwright" zamiast
//! „PlayVrit", „software house" zamiast „Softwarehouse" i „outsourcingowym"
//! zamiast „o utsourcingowym". Plik kontekstu już te nazwy ma, więc nie ma
//! powodu przepisywać ich ręcznie.
//!
//! Port z `Vocabulary.swift`.

use std::collections::HashMap;

use crate::text::{char_len, trim_ws, Text};

pub struct Vocabulary;

impl Vocabulary {
    /// Ile znaków słownictwa idzie do promptu whispera. Prompt ma ~224 tokeny
    /// i whisper.cpp przy nadmiarze ucina go od początku, czyli od słownictwa.
    pub const MAX_CHARS: usize = 350;

    /// Nazwy własne i technologie z tekstu, od najczęstszych.
    ///
    /// Bierzemy słowa, które wyglądają na nazwę: z wielką literą w środku
    /// zdania (Python, Playwright), z wielką literą w środku słowa
    /// (TypeScript), z kropką lub cyfrą (Next.js, B2B) albo skróty (DPP).
    /// Kod w backtickach pomijamy: identyfikatory ze ściągi (`defineModel`)
    /// w mowie nie padają, a zjadałyby limit.
    pub fn terms(context: &str) -> Vec<String> {
        let chars: Vec<char> = strip_code(context).chars().collect();
        let mut counts: HashMap<String, usize> = HashMap::new();
        let mut first_seen: HashMap<String, usize> = HashMap::new();
        let mut order = 0;
        let mut add = |term: String| {
            *counts.entry(term.clone()).or_insert(0) += 1;
            first_seen.entry(term).or_insert_with(|| {
                order += 1;
                order - 1
            });
        };

        // Sąsiednie nazwy rozdzielone jedną spacją to jedna nazwa:
        // „React Native", „Claude Code". Osobno „Native" nic whisperowi nie mówi.
        let mut phrase: Vec<String> = Vec::new();
        let mut phrase_end: isize = -1;
        let mut flush = |phrase: &mut Vec<String>| {
            if !phrase.is_empty() {
                add(phrase.join(" "));
            }
            phrase.clear();
        };

        let mut i = 0;
        while i < chars.len() {
            if !chars[i].is_alphabetic() {
                i += 1;
                continue;
            }
            let mut j = i;
            while j < chars.len()
                && (chars[j].is_alphabetic()
                    || chars[j].is_numeric()
                    || "+#".contains(chars[j])
                    || (".-".contains(chars[j]) && j + 1 < chars.len() && chars[j + 1].is_alphabetic()))
            {
                j += 1;
            }
            let word: String = chars[i..j].iter().collect();
            // Zdanie zaczynające się od nazwy („Python jest…") też ją liczy,
            // jeśli zaraz po niej idzie kolejna („Google Cloud").
            let continues = !phrase.is_empty() && phrase_end == i as isize - 1 && chars[i - 1] == ' ' && phrase.len() < 3;
            if is_term(&word, at_sentence_start(&chars, i) && !continues) {
                if !continues {
                    flush(&mut phrase);
                }
                phrase.push(word);
                phrase_end = j as isize;
            } else {
                flush(&mut phrase);
            }
            i = j;
        }
        flush(&mut phrase);

        let mut ranked: Vec<String> = counts.keys().cloned().collect();
        ranked.sort_by(|a, b| {
            tier(a)
                .cmp(&tier(b))
                .then_with(|| counts[b].cmp(&counts[a]))
                .then_with(|| first_seen[a].cmp(&first_seen[b]))
        });
        // Odmiany tej samej nazwy („Figma", „Figmy") zostają raz, w częstszej formie.
        let mut out: Vec<String> = Vec::new();
        for term in ranked {
            if !out.iter().any(|o| is_inflection(&term, o)) {
                out.push(term);
            }
        }
        out
    }

    /// Słownictwo z Ustawień, a po nim terminy z kontekstu, których jeszcze
    /// nie ma, do limitu znaków. Swift: domyślnie `max_chars = MAX_CHARS`.
    pub fn merge(user: &str, context: &str, max_chars: usize) -> String {
        let mut out = Text::normalize(user);
        if out.ends_with('.') {
            out.pop();
        }
        let mut known: Vec<String> = Vec::new();
        for entry in out.split([',', ';']).filter(|e| !e.is_empty()) {
            let phrase = trim_ws(entry);
            known.push(phrase.to_string());
            known.extend(phrase.split(' ').filter(|w| !w.is_empty()).map(String::from));
        }
        for term in Self::terms(context) {
            if known.iter().any(|k| k.to_lowercase() == term.to_lowercase() || is_inflection(&term, k)) {
                continue;
            }
            let next = if out.is_empty() { term.clone() } else { format!("{out}, {term}") };
            if char_len(&next) > max_chars {
                continue;
            }
            out = next;
            known.push(term);
        }
        out
    }
}

/// Kolejność przy ograniczonym miejscu: najpierw to, co wygląda na
/// technologię (TypeScript, Next.js, B2B), potem zwykłe nazwy (Python,
/// Playwright), na końcu słowa odmienione po polsku albo z polskimi
/// znakami („Brazylii", „Wrocław"). Whisper zna polskie słowa, a nazw
/// technologii bez podpowiedzi nie.
pub(crate) fn tier(term: &str) -> u8 {
    let word = term.split(' ').rfind(|w| !w.is_empty()).unwrap_or(term);
    // Identyfikatory z kodu („useState") w mowie padają rzadko.
    if word.chars().next().is_some_and(char::is_lowercase) && word.chars().any(char::is_uppercase) {
        return 2;
    }
    // Same wielkie litery („SEO", „AWS") whisper zwykle zna: jak zwykłe nazwy.
    let acronym = word.chars().all(|c| c.is_uppercase() || c.is_numeric()) && !word.chars().any(char::is_numeric);
    if !acronym && (word.chars().skip(1).any(char::is_uppercase) || word.contains('.') || word.chars().any(char::is_numeric)) {
        return 0;
    }
    let lower = word.to_lowercase();
    if lower.chars().any(|c| "ąćęłńóśźż".contains(c)) {
        return 2;
    }
    if ["ii", "ji", "ie", "ce", "owi", "ach", "ami", "ego", "emu", "ych", "ów"].iter().any(|s| lower.ends_with(s)) {
        return 2;
    }
    1
}

/// „Figmy" to odmiana „Figma", „Reacta" odmiana „React": ten sam rdzeń,
/// inna końcówka, najwyżej trzy znaki różnicy.
pub(crate) fn is_inflection(a: &str, b: &str) -> bool {
    let x: Vec<char> = a.to_lowercase().chars().collect();
    let y: Vec<char> = b.to_lowercase().chars().collect();
    if x == y || x.contains(&' ') || y.contains(&' ') || x.len() < 4 || y.len() < 4 || x.len().abs_diff(y.len()) > 3 {
        return false;
    }
    let stem = x.len().min(y.len()) - 1;
    stem >= 4 && x[..stem] == y[..stem]
}

fn strip_code(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_code = false;
    for ch in text.chars() {
        if ch == '`' {
            in_code = !in_code;
            out.push(' ');
            continue;
        }
        out.push(if in_code { ' ' } else { ch });
    }
    out
}

fn at_sentence_start(chars: &[char], index: usize) -> bool {
    let mut k = index as isize - 1;
    while k >= 0 && (chars[k as usize] == ' ' || chars[k as usize] == '*') {
        k -= 1;
    }
    k < 0 || ".!?:\n#(-\"„".contains(chars[k as usize])
}

fn is_term(word: &str, sentence_start: bool) -> bool {
    // „UI", „CV", „B2": krótkie skróty whisper zna i bez podpowiedzi.
    let n = char_len(word);
    if !(3..=30).contains(&n) {
        return false;
    }
    if !word.chars().any(char::is_alphabetic) {
        return false;
    }
    let first = word.chars().next().expect("niepuste");
    if word.chars().skip(1).any(char::is_uppercase) {
        return true; // TypeScript, DPP, B2B
    }
    // Next.js, example.com; ale nie „m.in" ani „np".
    if word.contains('.') && word.split('.').filter(|p| !p.is_empty()).all(|p| char_len(p) >= 2) {
        return true;
    }
    if word.chars().any(char::is_numeric) && first.is_uppercase() {
        return true;
    }
    // Wielka litera w środku zdania: nazwa własna (Python, Playwright).
    first.is_uppercase() && !sentence_start
}
