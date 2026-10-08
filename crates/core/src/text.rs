//! Scalanie strumieniowego tekstu z rozpoznawania na żywo.
//!
//! Silniki ASR aktualizują ten sam blok tekstu w miejscu: tekst rośnie, bywa
//! poprawiany, a czasem przycinany od początku (przewijane okno). `reconcile`
//! sprowadza kolejne migawki do jednego, niezduplikowanego zdania.
//!
//! Port z `Text.swift` (a ten 1:1 z `extension/src/core/text.js`). Operujemy na
//! `char`, a nie na bajtach — inaczej polskie znaki rozjechałyby indeksy przy
//! liczeniu wspólnego prefiksu.

use std::sync::OnceLock;

use regex::Regex;

pub struct Text;

const ZERO_WIDTH: [char; 5] = ['\u{200B}', '\u{200C}', '\u{200D}', '\u{FEFF}', '\u{00AD}'];

/// Minimalna liczba znaków nakładki, przy której ufamy sklejeniu.
const MIN_OVERLAP: usize = 6;
/// Minimalny wspólny prefiks, przy którym uznajemy migawkę za poprawkę.
const MIN_REVISION_PREFIX: usize = 12;

/// Odpowiednik `CharacterSet.whitespaces` ze Swifta: białe znaki poziome,
/// bez łamań linii.
pub(crate) fn is_horizontal_ws(c: char) -> bool {
    c.is_whitespace() && !matches!(c, '\n' | '\r' | '\u{0B}' | '\u{0C}' | '\u{85}' | '\u{2028}' | '\u{2029}')
}

/// `trimmingCharacters(in: .whitespaces)`.
pub(crate) fn trim_ws(s: &str) -> &str {
    s.trim_matches(is_horizontal_ws)
}

/// Liczba znaków (Swift `String.count`).
pub(crate) fn char_len(s: &str) -> usize {
    s.chars().count()
}

/// Pierwsze `n` znaków (Swift `prefix(n)`).
pub(crate) fn prefix_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

/// Ostatnie `n` znaków (Swift `suffix(n)`).
pub(crate) fn suffix_chars(s: &str, n: usize) -> String {
    let len = char_len(s);
    s.chars().skip(len.saturating_sub(n)).collect()
}

fn hallucinations() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    // Oryginał ma przed pierwszą gałęzią lookbehind `(?:^|(?<=[\s.,!?]))`,
    // którego crate `regex` nie obsługuje — sprawdzamy go ręcznie w
    // `strip_hallucinations`.
    RE.get_or_init(|| {
        Regex::new(
            r"(?i)(?:z?dzi[eę]kuj[eę]|zdj[eę]kuj[eę]|dzi[eę]ki)\s+(?:bardzo\s+)?za\s+(?:uwag[eę]|ogl[aą]danie|obejrzenie)[.!]*|napisy\s+(?:stworzone|wykonane)\s+przez[^.!?]*[.!?]?",
        )
        .expect("regex halucynacji")
    })
}

impl Text {
    /// Normalizacja białych znaków + usunięcie znaków zerowej szerokości.
    pub fn normalize(input: &str) -> String {
        let mut out = String::with_capacity(input.len());
        let mut pending_space = false;
        let mut started = false;
        for ch in input.chars() {
            if ZERO_WIDTH.contains(&ch) {
                continue;
            }
            if ch.is_whitespace() {
                if started {
                    pending_space = true;
                }
                continue;
            }
            if pending_space {
                out.push(' ');
                pending_space = false;
            }
            out.push(ch);
            started = true;
        }
        out
    }

    /// Długość wspólnego prefiksu dwóch napisów (w znakach).
    pub fn common_prefix_length(a: &[char], b: &[char]) -> usize {
        let max = a.len().min(b.len());
        let mut i = 0;
        while i < max && a[i] == b[i] {
            i += 1;
        }
        i
    }

    /// Najdłuższy sufiks `a`, który jest prefiksem `b`.
    /// Okno ograniczone do `limit` znaków — chroni przed O(n²) na długich blokach.
    pub fn overlap_length(a: &[char], b: &[char], limit: usize) -> usize {
        let max = a.len().min(b.len()).min(limit);
        let mut k = max;
        while k > 0 {
            if a[a.len() - k..] == b[..k] {
                return k;
            }
            k -= 1;
        }
        0
    }

    /// Łączy poprzedni stan segmentu z nową migawką tekstu.
    pub fn reconcile(prev: &str, next: &str) -> String {
        let a = Self::normalize(prev);
        let b = Self::normalize(next);
        if a.is_empty() {
            return b;
        }
        if b.is_empty() {
            return a;
        }
        if a == b {
            return a;
        }

        // 1. Wzrost strumienia: nowa migawka to stara + ogon.
        if b.starts_with(&a) {
            return b;
        }

        let ac: Vec<char> = a.chars().collect();
        let bc: Vec<char> = b.chars().collect();

        // 2. Poprawka in-place: wspólny początek, zmieniona końcówka.
        let cp = Self::common_prefix_length(&ac, &bc);
        if cp >= MIN_REVISION_PREFIX || (!ac.is_empty() && cp as f64 >= ac.len() as f64 * 0.6) {
            return if bc.len() >= ac.len() { b } else { a };
        }

        // 3. Przewijane okno: koniec starego = początek nowego.
        let k = Self::overlap_length(&ac, &bc, 240);
        if k >= MIN_OVERLAP {
            return a + &bc[k..].iter().collect::<String>();
        }

        // 4. Zawieranie — nic nowego albo pełne rozszerzenie.
        if a.contains(&b) {
            return a;
        }
        if b.contains(&a) {
            return b;
        }

        // 5. Rozłączne fragmenty tej samej wypowiedzi — doklejamy.
        format!("{a} {b}")
    }

    /// Zgrubna liczba słów (do heurystyk i statystyk).
    pub fn word_count(text: &str) -> usize {
        Self::normalize(text).split(' ').filter(|w| !w.is_empty()).count()
    }

    /// Ucina tekst do `max` znaków, dodając wielokropek (domyślnie 120).
    pub fn truncate(text: &str, max: usize) -> String {
        let t = Self::normalize(text);
        if char_len(&t) <= max {
            t
        } else {
            prefix_chars(&t, max.saturating_sub(1)) + "…"
        }
    }

    /// whisper.cpp zwraca tekst z twardymi łamaniami linii i znacznikami
    /// nie-mowy — w transkrypcie chcemy jedną, czystą linię.
    ///
    /// Bez tego `[BLANK_AUDIO]` i `(szum)` trafiałyby do notatki jako
    /// wypowiedzi, a whisper dokleja je chętnie na ciszy.
    pub fn clean_whisper(raw: &str) -> String {
        let mut out = String::with_capacity(raw.len());
        let mut square = 0i32;
        let mut round = 0i32;
        let chars: Vec<char> = raw.chars().collect();
        for (i, &ch) in chars.iter().enumerate() {
            // Granica segmentu w środku słowa: litera, łamanie, mała litera bez
            // spacji (nowe słowo whisper zaczyna od spacji). Bez tego
            // „odpow\niedzialny" dawało w notatce „odpow iedzialny".
            if ch == '\n'
                && i > 0
                && i + 1 < chars.len()
                && (chars[i - 1].is_alphabetic() || chars[i - 1].is_numeric())
                && chars[i + 1].is_lowercase()
            {
                continue;
            }
            match ch {
                '[' => {
                    square += 1;
                    out.push(' ');
                }
                ']' => {
                    square = (square - 1).max(0);
                    out.push(' ');
                }
                '(' => {
                    round += 1;
                    out.push(' ');
                }
                ')' => {
                    round = (round - 1).max(0);
                    out.push(' ');
                }
                _ => {
                    if square == 0 && round == 0 {
                        out.push(ch);
                    }
                }
            }
        }
        Self::strip_hallucinations(&out)
    }

    /// Usuwa znane halucynacje whispera na ciszy i szumie („Dziękuję za uwagę.",
    /// „Zdjękuje za oglądanie!"). W rozmowie padały co minutę, a doklejone do
    /// pytania psuły prompt.
    pub fn strip_hallucinations(text: &str) -> String {
        let re = hallucinations();
        let mut out = String::with_capacity(text.len());
        let mut last = 0;
        let mut pos = 0;
        while let Some(m) = re.find_at(text, pos) {
            let starts_napisy = m.as_str().chars().next().is_some_and(|c| c == 'n' || c == 'N');
            // Lookbehind z oryginału: gałąź „dziękuję…" tylko na początku tekstu
            // albo po białym znaku / interpunkcji.
            let anchored = starts_napisy
                || m.start() == 0
                || text[..m.start()]
                    .chars()
                    .next_back()
                    .is_some_and(|c| c.is_whitespace() || ".,!?".contains(c));
            if !anchored {
                let step = text[m.start()..].chars().next().map_or(1, char::len_utf8);
                pos = m.start() + step;
                continue;
            }
            out.push_str(&text[last..m.start()]);
            out.push(' ');
            last = m.end();
            pos = m.end();
        }
        out.push_str(&text[last..]);
        Self::normalize(&out)
    }

    /// Składa tekst do ASCII: małe litery, bez znaków diakrytycznych.
    ///
    /// Bez tego wzorce są kruche: w mowie szyk jest swobodny, a `ł` nie rozkłada
    /// się przez NFD na literę + znak łączący, więc wymaga osobnego podstawienia.
    pub fn fold(text: &str) -> String {
        let lowered = text.to_lowercase().replace('ł', "l");
        lowered
            .chars()
            .filter(|c| !('\u{0300}'..='\u{036F}').contains(c))
            .map(|c| match DIACRITIC_FOLD.binary_search_by_key(&c, |&(k, _)| k) {
                Ok(i) => DIACRITIC_FOLD[i].1,
                Err(_) => c,
            })
            .collect()
    }
}

/// Rozkład kanoniczny (NFD) bez znaków łączących, dla liter łacińskich małych.
/// Wygenerowane z unicodedata; posortowane po kodzie znaku.
static DIACRITIC_FOLD: &[(char, char)] = &[
    ('\u{E0}', 'a'), ('\u{E1}', 'a'), ('\u{E2}', 'a'), ('\u{E3}', 'a'), ('\u{E4}', 'a'), ('\u{E5}', 'a'),
    ('\u{E7}', 'c'), ('\u{E8}', 'e'), ('\u{E9}', 'e'), ('\u{EA}', 'e'), ('\u{EB}', 'e'), ('\u{EC}', 'i'),
    ('\u{ED}', 'i'), ('\u{EE}', 'i'), ('\u{EF}', 'i'), ('\u{F1}', 'n'), ('\u{F2}', 'o'), ('\u{F3}', 'o'),
    ('\u{F4}', 'o'), ('\u{F5}', 'o'), ('\u{F6}', 'o'), ('\u{F9}', 'u'), ('\u{FA}', 'u'), ('\u{FB}', 'u'),
    ('\u{FC}', 'u'), ('\u{FD}', 'y'), ('\u{FF}', 'y'), ('\u{101}', 'a'), ('\u{103}', 'a'), ('\u{105}', 'a'),
    ('\u{107}', 'c'), ('\u{109}', 'c'), ('\u{10B}', 'c'), ('\u{10D}', 'c'), ('\u{10F}', 'd'), ('\u{113}', 'e'),
    ('\u{115}', 'e'), ('\u{117}', 'e'), ('\u{119}', 'e'), ('\u{11B}', 'e'), ('\u{11D}', 'g'), ('\u{11F}', 'g'),
    ('\u{121}', 'g'), ('\u{123}', 'g'), ('\u{125}', 'h'), ('\u{129}', 'i'), ('\u{12B}', 'i'), ('\u{12D}', 'i'),
    ('\u{12F}', 'i'), ('\u{135}', 'j'), ('\u{137}', 'k'), ('\u{13A}', 'l'), ('\u{13C}', 'l'), ('\u{13E}', 'l'),
    ('\u{144}', 'n'), ('\u{146}', 'n'), ('\u{148}', 'n'), ('\u{14D}', 'o'), ('\u{14F}', 'o'), ('\u{151}', 'o'),
    ('\u{155}', 'r'), ('\u{157}', 'r'), ('\u{159}', 'r'), ('\u{15B}', 's'), ('\u{15D}', 's'), ('\u{15F}', 's'),
    ('\u{161}', 's'), ('\u{163}', 't'), ('\u{165}', 't'), ('\u{169}', 'u'), ('\u{16B}', 'u'), ('\u{16D}', 'u'),
    ('\u{16F}', 'u'), ('\u{171}', 'u'), ('\u{173}', 'u'), ('\u{175}', 'w'), ('\u{177}', 'y'), ('\u{17A}', 'z'),
    ('\u{17C}', 'z'), ('\u{17E}', 'z'), ('\u{1A1}', 'o'), ('\u{1B0}', 'u'), ('\u{1CE}', 'a'), ('\u{1D0}', 'i'),
    ('\u{1D2}', 'o'), ('\u{1D4}', 'u'), ('\u{1D6}', 'u'), ('\u{1D8}', 'u'), ('\u{1DA}', 'u'), ('\u{1DC}', 'u'),
    ('\u{1DF}', 'a'), ('\u{1E1}', 'a'), ('\u{1E3}', 'æ'), ('\u{1E7}', 'g'), ('\u{1E9}', 'k'), ('\u{1EB}', 'o'),
    ('\u{1ED}', 'o'), ('\u{1EF}', 'ʒ'), ('\u{1F0}', 'j'), ('\u{1F5}', 'g'), ('\u{1F9}', 'n'), ('\u{1FB}', 'a'),
    ('\u{1FD}', 'æ'), ('\u{1FF}', 'ø'), ('\u{201}', 'a'), ('\u{203}', 'a'), ('\u{205}', 'e'), ('\u{207}', 'e'),
    ('\u{209}', 'i'), ('\u{20B}', 'i'), ('\u{20D}', 'o'), ('\u{20F}', 'o'), ('\u{211}', 'r'), ('\u{213}', 'r'),
    ('\u{215}', 'u'), ('\u{217}', 'u'), ('\u{219}', 's'), ('\u{21B}', 't'), ('\u{21F}', 'h'), ('\u{227}', 'a'),
    ('\u{229}', 'e'), ('\u{22B}', 'o'), ('\u{22D}', 'o'), ('\u{22F}', 'o'), ('\u{231}', 'o'), ('\u{233}', 'y'),
    ('\u{1E01}', 'a'), ('\u{1E03}', 'b'), ('\u{1E05}', 'b'), ('\u{1E07}', 'b'), ('\u{1E09}', 'c'), ('\u{1E0B}', 'd'),
    ('\u{1E0D}', 'd'), ('\u{1E0F}', 'd'), ('\u{1E11}', 'd'), ('\u{1E13}', 'd'), ('\u{1E15}', 'e'), ('\u{1E17}', 'e'),
    ('\u{1E19}', 'e'), ('\u{1E1B}', 'e'), ('\u{1E1D}', 'e'), ('\u{1E1F}', 'f'), ('\u{1E21}', 'g'), ('\u{1E23}', 'h'),
    ('\u{1E25}', 'h'), ('\u{1E27}', 'h'), ('\u{1E29}', 'h'), ('\u{1E2B}', 'h'), ('\u{1E2D}', 'i'), ('\u{1E2F}', 'i'),
    ('\u{1E31}', 'k'), ('\u{1E33}', 'k'), ('\u{1E35}', 'k'), ('\u{1E37}', 'l'), ('\u{1E39}', 'l'), ('\u{1E3B}', 'l'),
    ('\u{1E3D}', 'l'), ('\u{1E3F}', 'm'), ('\u{1E41}', 'm'), ('\u{1E43}', 'm'), ('\u{1E45}', 'n'), ('\u{1E47}', 'n'),
    ('\u{1E49}', 'n'), ('\u{1E4B}', 'n'), ('\u{1E4D}', 'o'), ('\u{1E4F}', 'o'), ('\u{1E51}', 'o'), ('\u{1E53}', 'o'),
    ('\u{1E55}', 'p'), ('\u{1E57}', 'p'), ('\u{1E59}', 'r'), ('\u{1E5B}', 'r'), ('\u{1E5D}', 'r'), ('\u{1E5F}', 'r'),
    ('\u{1E61}', 's'), ('\u{1E63}', 's'), ('\u{1E65}', 's'), ('\u{1E67}', 's'), ('\u{1E69}', 's'), ('\u{1E6B}', 't'),
    ('\u{1E6D}', 't'), ('\u{1E6F}', 't'), ('\u{1E71}', 't'), ('\u{1E73}', 'u'), ('\u{1E75}', 'u'), ('\u{1E77}', 'u'),
    ('\u{1E79}', 'u'), ('\u{1E7B}', 'u'), ('\u{1E7D}', 'v'), ('\u{1E7F}', 'v'), ('\u{1E81}', 'w'), ('\u{1E83}', 'w'),
    ('\u{1E85}', 'w'), ('\u{1E87}', 'w'), ('\u{1E89}', 'w'), ('\u{1E8B}', 'x'), ('\u{1E8D}', 'x'), ('\u{1E8F}', 'y'),
    ('\u{1E91}', 'z'), ('\u{1E93}', 'z'), ('\u{1E95}', 'z'), ('\u{1E96}', 'h'), ('\u{1E97}', 't'), ('\u{1E98}', 'w'),
    ('\u{1E99}', 'y'), ('\u{1E9B}', 'ſ'), ('\u{1EA1}', 'a'), ('\u{1EA3}', 'a'), ('\u{1EA5}', 'a'), ('\u{1EA7}', 'a'),
    ('\u{1EA9}', 'a'), ('\u{1EAB}', 'a'), ('\u{1EAD}', 'a'), ('\u{1EAF}', 'a'), ('\u{1EB1}', 'a'), ('\u{1EB3}', 'a'),
    ('\u{1EB5}', 'a'), ('\u{1EB7}', 'a'), ('\u{1EB9}', 'e'), ('\u{1EBB}', 'e'), ('\u{1EBD}', 'e'), ('\u{1EBF}', 'e'),
    ('\u{1EC1}', 'e'), ('\u{1EC3}', 'e'), ('\u{1EC5}', 'e'), ('\u{1EC7}', 'e'), ('\u{1EC9}', 'i'), ('\u{1ECB}', 'i'),
    ('\u{1ECD}', 'o'), ('\u{1ECF}', 'o'), ('\u{1ED1}', 'o'), ('\u{1ED3}', 'o'), ('\u{1ED5}', 'o'), ('\u{1ED7}', 'o'),
    ('\u{1ED9}', 'o'), ('\u{1EDB}', 'o'), ('\u{1EDD}', 'o'), ('\u{1EDF}', 'o'), ('\u{1EE1}', 'o'), ('\u{1EE3}', 'o'),
    ('\u{1EE5}', 'u'), ('\u{1EE7}', 'u'), ('\u{1EE9}', 'u'), ('\u{1EEB}', 'u'), ('\u{1EED}', 'u'), ('\u{1EEF}', 'u'),
    ('\u{1EF1}', 'u'), ('\u{1EF3}', 'y'), ('\u{1EF5}', 'y'), ('\u{1EF7}', 'y'), ('\u{1EF9}', 'y'),
];
