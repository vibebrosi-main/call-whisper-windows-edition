//! Logika asystenta: co jest pytaniem i jak zbudować z transkrypcji prompt.
//! Czysta i testowalna — nie wie nic o HTTP ani o dostawcy modelu.
//! Port z `Assistant.swift` (a ten z `extension/src/core/assistant.js`).

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::text::{char_len, prefix_chars, suffix_chars, trim_ws, Text};
use crate::transcript_store::Segment;

/// Słowa otwierające pytanie. Pytajnika w mowie nie ma — ASR go nie zawsze daje.
const QUESTION_OPENERS_PL: &[&str] = &[
    "czy", "jak", "jaki", "jaka", "jakie", "jakim", "jakich", "ile", "kiedy", "gdzie",
    "kto", "komu", "kogo", "co", "czemu", "dlaczego", "po co", "skad", "dokad", "ktory",
    "ktora", "ktore", "czym", "w czym", "na czym",
    // Formy odmienione: bez nich „od której wersji" i „jakiego typu" przepadały.
    "której", "którego", "którym", "których", "jakiego", "jakiej", "jakimi", "ilu",
];
const QUESTION_OPENERS_EN: &[&str] = &[
    "is", "are", "was", "were", "do", "does", "did", "can", "could", "should", "would",
    "will", "what", "why", "how", "when", "where", "who", "which", "whose", "whom",
];

fn all_openers() -> &'static Vec<Vec<String>> {
    static V: OnceLock<Vec<Vec<String>>> = OnceLock::new();
    V.get_or_init(|| {
        QUESTION_OPENERS_PL
            .iter()
            .chain(QUESTION_OPENERS_EN)
            .map(|o| Text::fold(o).split(' ').filter(|w| !w.is_empty()).map(String::from).collect())
            .collect()
    })
}

/// Zwroty, po których ktoś prosi o konkret — szukane w dowolnym miejscu.
/// Zapisane bez znaków diakrytycznych, bo porównujemy tekst złożony do ASCII.
const ASK_PHRASES: &[&str] = &[
    "mam pytanie", "pytanie do", "wie ktos", "wiesz moze", "czy ktos wie",
    "jak to dziala", "co to znaczy", "zastanawiam sie", "nie wiem czy",
    "ciekawi mnie", "wytlumacz", "przypomnij mi",
    // Rozmowa rekrutacyjna to w połowie polecenia, nie pytania: „opowiedz mi
    // o projekcie", „przybliż, za co odpowiadałeś". Bez pytajnika i bez słowa
    // pytającego na początku przepadały wszystkie.
    "opowiedz", "opowiesz", "opisz", "przybliz", "podziel sie", "pochwal sie",
    "powiedz mi o", "powiedz cos o", "dlaczego", "od kiedy", "jak wyglada",
    "jesli mialbys", "jesli mialabys", "gdybys mial", "gdybys miala", "zgadza sie",
    // Prośba o potwierdzenie warunków: „…do końca roku jest dla ciebie okej."
    "dla ciebie ok", "dla ciebie okej", "pasuje ci", "odpowiada ci", "ci pasuje", "ci odpowiada",
    // Zaproszenie do pytań: „jeżeli masz jakieś pytania, śmiało". Podpowiedzią
    // są wtedy pytania do zadania, nie odpowiedź.
    "masz jakies", "masz pytania", "macie jakies pytania", "chcialbys zadac",
    "chcialbys zapytac", "chcesz o cos zapytac", "any questions",
    "tell me about", "walk me through", "describe",
    "anyone know", "does anyone", "what does", "how do we", "quick question",
];

/// Pytania techniczno-organizacyjne, na które asystent nie ma czego odpowiedzieć.
fn small_talk_patterns() -> &'static Vec<Regex> {
    static V: OnceLock<Vec<Regex>> = OnceLock::new();
    V.get_or_init(|| {
        [
            r"\b(slychac|slyszysz|slyszycie|slysze|slyszymy)\b",
            // „Widać mój ekran?" tak, „Gdzie się widzisz za pięć lat?" nie.
            r"\b(widac|widzisz|widzicie|widze)\b.*\b(mnie|nas|cie|was|ekran\w*|prezentacj\w*|kamer\w*)\b",
            r"\b(hear|see)\s+(me|my\s+screen|you)\b",
            r"\bhalo+\b",
            r"\bjestes\s+tam\b",
            r"\b(mozemy|to)\s+zaczyna(my|c)\b",
            r"\bwszyscy\s+(sa|juz\s+sa)\b",
            r"\b(dziala|dziela)\s+(mikrofon|kamera|dzwiek)\b",
        ]
        .iter()
        .filter_map(|p| Regex::new(p).ok())
        .collect()
    })
}

const MIN_QUESTION_CHARS: usize = 8;
const MIN_QUESTION_WORDS: usize = 3;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QuestionVerdict {
    pub is_question: bool,
    pub confidence: f64,
    pub reason: String,
    pub question: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Sentence {
    pub text: String,
    /// `?`, `.`, `!`, `…` (urwane) albo pusty (bez interpunkcji).
    pub end: String,
}

/// Twardy limit długości pytania. Prompt ma być krótki, nie kompletny.
const MAX_QUESTION_CHARS: usize = 220;
/// Pytanie krótsze niż to samo nic nie znaczy („Zgadza się?", „Jakie
/// były?") i dostaje zdanie, które stoi przed nim.
const MIN_STANDALONE_CHARS: usize = 30;

/// Słówka, które w mowie stoją PRZED właściwym słowem pytającym: „od której
/// wersji", „w czym piszesz". Dopuszczamy najwyżej jedno.
const LEADING_PARTICLES: &[&str] = &["od", "do", "w", "na", "z", "za", "po", "przy", "dla", "o", "u"];

/// Słowa wypełniające i wstępy, po których dopiero zaczyna się treść:
/// „Okej, dobra, a powiedz mi, jak długo programujesz?". Fraza złożona
/// wyłącznie z nich jest wstępem, a nie treścią.
const FILLER_WORDS: &[&str] = &[
    "okej", "ok", "okay", "dobra", "dobrze", "no", "tak", "mhm", "hmm", "ehm", "eh", "yyy", "aha",
    "fajnie", "super", "jasne", "swietnie", "sluchaj", "wiesz", "czekaj", "hej",
    "a", "i", "to", "wiec", "ale", "czyli", "jeszcze", "jedno", "jakby", "teraz",
    "powiedz", "powiedzcie", "mi", "nam", "mam", "pytanie", "pytanko", "w", "sensie", "znaczy",
    "na", "przyklad", "generalnie", "ogolnie", "mozesz", "mozecie", "powiedziec",
    // „Nie, po prostu jakie…" - sprzeciw przed właściwym pytaniem.
    "nie",
    "so", "well", "alright", "right", "tell", "me", "question",
];

/// Końcówki, które z oznajmienia robią prośbę o potwierdzenie: „…, tak?".
const CONFIRMATION_TAGS: &[&str] = &["tak", "nie", "prawda", "no nie", "nie prawda", "right", "yeah"];

/// „Jak widzisz, …", „jak wiesz, …" - wtrącenia, nie pytania.
const ASIDE_VERBS: &[&str] = &[
    "widzisz", "wiesz", "slyszysz", "rozumiesz", "mowisz", "pamietasz", "mowiles", "wspominales",
];

fn is_filler(w: &str) -> bool {
    FILLER_WORDS.contains(&w)
}

#[derive(Debug, Clone)]
struct Clause {
    text: String,
    start: usize,
}

#[derive(Debug, Clone)]
struct Score {
    confidence: f64,
    reasons: Vec<String>,
    start: usize,
}

pub struct QuestionDetector;

impl QuestionDetector {
    /// Dzieli wypowiedź na frazy po interpunkcji, którą daje ASR.
    pub fn split_clauses(text: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut buffer = String::new();
        for ch in text.chars() {
            if ",.;!?".contains(ch) {
                let trimmed = trim_ws(&buffer);
                if !trimmed.is_empty() {
                    out.push(trimmed.to_string());
                }
                buffer.clear();
                continue;
            }
            buffer.push(ch);
        }
        let tail = trim_ws(&buffer);
        if !tail.is_empty() {
            out.push(tail.to_string());
        }
        out
    }

    /// Dzieli wypowiedź na zdania, pamiętając, czym się kończyły.
    ///
    /// Kropka kończy zdanie tylko przed spacją albo na końcu: „Next.js" i „40 ml."
    /// to nie są dwa zdania. Dwie kropki i więcej to wielokropek.
    pub fn split_sentences(text: &str) -> Vec<Sentence> {
        fn push(out: &mut Vec<Sentence>, buffer: &mut String, end: &str) {
            let trimmed = trim_ws(buffer);
            if !trimmed.is_empty() {
                out.push(Sentence { text: trimmed.to_string(), end: end.to_string() });
            }
            buffer.clear();
        }
        let mut out = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut buffer = String::new();
        let mut i = 0;
        while i < chars.len() {
            let ch = chars[i];
            if ch == '?' || ch == '!' {
                let mut end = ch;
                while i + 1 < chars.len() && (chars[i + 1] == '?' || chars[i + 1] == '!') {
                    if chars[i + 1] == '?' {
                        end = '?';
                    }
                    i += 1;
                }
                push(&mut out, &mut buffer, &end.to_string());
            } else if ch == '…' {
                push(&mut out, &mut buffer, "…");
            } else if ch == '.' {
                let mut run = 1;
                while i + run < chars.len() && chars[i + run] == '.' {
                    run += 1;
                }
                if run >= 2 {
                    i += run;
                    push(&mut out, &mut buffer, "…");
                    continue;
                }
                if i + 1 == chars.len() || chars[i + 1].is_whitespace() {
                    push(&mut out, &mut buffer, ".");
                } else {
                    buffer.push(ch);
                }
            } else {
                buffer.push(ch);
            }
            i += 1;
        }
        push(&mut out, &mut buffer, "");
        out
    }

    fn words(text: &str) -> Vec<String> {
        Text::fold(text)
            .split(|c: char| !(c.is_ascii_alphanumeric()) && c != '\'')
            .filter(|w| !w.is_empty())
            .map(String::from)
            .collect()
    }

    fn is_second_person(word: &str) -> bool {
        if !word.chars().all(char::is_alphabetic) || ASIDE_VERBS.contains(&word) {
            return false;
        }
        if word == "jestes" {
            return true;
        }
        let n = char_len(word);
        ["les", "las", "lbys", "labys", "esz", "isz", "ysz", "asz"]
            .iter()
            .any(|s| word.ends_with(s) && n >= s.len() + 3)
    }

    fn is_filler_clause(clause: &str) -> bool {
        Self::words(clause).iter().all(|w| is_filler(w))
    }

    fn has_ask_phrase(text: &str) -> bool {
        let folded = Self::words(text).join(" ");
        ASK_PHRASES.iter().any(|p| folded.contains(p))
    }

    /// Czy fraza zaczyna się od słowa pytającego (po wypełniaczach i jednym przyimku).
    fn opens_question(clause: &str) -> bool {
        let openers = all_openers();
        let mut all = Self::words(clause);
        while let Some(first) = all.first().cloned() {
            // „Nie, po prostu jakie featury…" - „po" otwiera też „po co", więc
            // „po prostu" zdejmujemy jako parę.
            if first == "po" && all.len() > 1 && all[1] == "prostu" {
                all.drain(..2);
            } else if is_filler(&first) && !openers.iter().any(|o| o[0] == first) {
                all.remove(0);
            } else {
                break;
            }
        }
        let Some(first) = all.first() else { return false };
        let mut starts: Vec<&[String]> = vec![&all[..]];
        if LEADING_PARTICLES.contains(&first.as_str()) {
            starts.push(&all[1..]);
        }
        starts.iter().any(|ws| {
            !ws.is_empty()
                && openers
                    .iter()
                    .any(|parts| parts.len() <= ws.len() && parts.iter().enumerate().all(|(i, p)| &ws[i] == p))
        })
    }

    /// Frazy zdania razem z miejscem (w znakach), w którym zaczynają się w tekście.
    fn clauses(sentence: &str) -> Vec<Clause> {
        let chars: Vec<char> = sentence.chars().collect();
        let mut out = Vec::new();
        let mut start = 0;
        for i in 0..=chars.len() {
            if i == chars.len() || chars[i] == ',' || chars[i] == ';' {
                let s: String = chars[start..i].iter().collect();
                let text = trim_ws(&s);
                if !text.is_empty() {
                    out.push(Clause { text: text.to_string(), start });
                }
                start = i + 1;
            }
        }
        out
    }

    /// Ocena jednego zdania.
    ///
    /// Słowo pytające liczy się tylko na początku zdania: po wypełniaczach
    /// („Okej, dobra, a jak…") albo po wstępie („mam pytanie, czym…",
    /// „powiedz mi, ile…"). Po zwykłej frazie to prawie zawsze zaimek względny
    /// albo spójnik: „Podobało mi się, jak zrobiłeś", „praca, która była".
    fn score(sentence: &Sentence) -> Score {
        let parts = Self::clauses(&sentence.text);
        let first: isize = parts.iter().position(|c| !Self::is_filler_clause(&c.text)).map_or(-1, |i| i as isize);
        let mut head = first;
        let mut opener = false;
        let mut k = first.max(0) as usize;
        while k < parts.len() {
            let at_start = k as isize == first;
            let after_lead =
                k > 0 && (Self::is_filler_clause(&parts[k - 1].text) || Self::has_ask_phrase(&parts[k - 1].text));
            if (at_start || after_lead) && Self::opens_question(&parts[k].text) {
                opener = true;
                // „Opowiedz mi o projekcie, czym się tam zajmowałeś" - polecenie
                // przed słowem pytającym niesie treść (o który projekt chodzi),
                // więc pytanie zaczyna się od niego. Sam wstęp („mam pytanie,
                // czym…") nie.
                let lead = k > 0
                    && !Self::is_filler_clause(&parts[k - 1].text)
                    && Self::has_ask_phrase(&parts[k - 1].text);
                head = if lead { k as isize - 1 } else { k as isize };
                break;
            }
            k += 1;
        }

        let asked = sentence.end == "?";
        let last_words = if parts.len() > 1 { Self::words(&parts[parts.len() - 1].text).join(" ") } else { String::new() };
        let confirmation = asked && !opener && CONFIRMATION_TAGS.contains(&last_words.as_str());
        let phrase = Self::has_ask_phrase(&sentence.text);

        let mut confidence = 0.0;
        let mut reasons: Vec<String> = Vec::new();
        if asked {
            confidence += if confirmation { 0.25 } else { 0.6 };
            reasons.push(if confirmation { "potwierdzenie" } else { "pytajnik" }.into());
        }
        if opener {
            // Kropka albo wykrzyknik od ASR to sygnał, że zdanie jest
            // oznajmujące, a wielokropek, że urwane.
            confidence += if asked || sentence.end.is_empty() { 0.4 } else { 0.2 };
            reasons.push("słowo-pytające".into());
            // Czasownik w 2. osobie („stworzyłeś") - zdanie jest skierowane do
            // rozmówcy. Z kropką od ASR słowo pytające samo nie wystarcza, ale
            // razem z takim czasownikiem to pytanie. Urwane („…ile już...") nie.
            if sentence.end == "." && Self::words(&sentence.text).iter().any(|w| Self::is_second_person(w)) {
                confidence += 0.2;
                reasons.push("do-rozmówcy".into());
            }
        }
        if phrase {
            confidence += 0.4;
            reasons.push("zwrot-pytający".into());
        }
        Score {
            confidence: f64::min(1.0, (confidence * 100.0_f64).round() / 100.0),
            reasons,
            start: if head >= 0 { parts[head as usize].start } else { 0 },
        }
    }

    /// Czy wypowiedź zawiera pytanie wymagające odpowiedzi merytorycznej.
    ///
    /// Oceniamy każde zdanie osobno i bierzemy najlepsze. Do modelu idzie samo
    /// pytanie: od jego początku (bez wstępu) do końca serii pytań po nim.
    pub fn detect(text: &str) -> QuestionVerdict {
        let raw = Text::strip_hallucinations(text);
        let folded = Text::fold(&raw);
        let all_words = folded.split(' ').filter(|w| !w.is_empty()).count();

        let no = |reason: &str| QuestionVerdict { is_question: false, confidence: 0.0, reason: reason.into(), question: String::new() };

        if char_len(&raw) < MIN_QUESTION_CHARS || all_words < MIN_QUESTION_WORDS {
            return no("za-krótkie");
        }
        if small_talk_patterns().iter().any(|re| re.is_match(&folded)) {
            return no("small-talk");
        }

        let sentences = Self::split_sentences(&raw);
        let mut scored: Vec<Score> = sentences.iter().map(Self::score).collect();
        if scored.is_empty() {
            return no("brak-sygnałów");
        }
        let mut best = 0;
        for (i, s) in scored.iter().enumerate() {
            if s.confidence > scored[best].confidence {
                best = i;
            }
        }
        let top = scored[best].clone();
        if top.confidence < 0.35 {
            return QuestionVerdict {
                is_question: false,
                confidence: top.confidence,
                reason: if top.reasons.is_empty() { "brak-sygnałów".into() } else { top.reasons.join("+") },
                question: String::new(),
            };
        }

        // Seria pytań wokół najlepszego; wstecz także potwierdzenia, bo bez
        // „…, której nie ma w CV, tak?" pytanie „czy ona gdzieś jest" nic nie znaczy.
        let mut from = best;
        while from > 0 && (scored[from - 1].confidence >= 0.35 || sentences[from - 1].end == "?") {
            from -= 1;
        }
        let mut to = best;
        while to + 1 < sentences.len() && sentences[to + 1].end == "?" && scored[to + 1].confidence >= 0.35 {
            to += 1;
        }

        let render = |scored: &[Score], a: usize, b: usize| -> String {
            (a..=b)
                .map(|i| {
                    let s = &sentences[i];
                    let body = if i == a {
                        let tail: String = s.text.chars().skip(scored[a].start).collect();
                        trim_ws(&tail).to_string()
                    } else {
                        s.text.clone()
                    };
                    body + &s.end
                })
                .collect::<Vec<_>>()
                .join(" ")
        };
        if from > 0 && char_len(&render(&scored, from, to)) < MIN_STANDALONE_CHARS {
            from -= 1;
            scored[from].start = 0;
        }
        let mut question = render(&scored, from, to);
        if char_len(&question) > MAX_QUESTION_CHARS {
            question = render(&scored, best, best);
        }
        if char_len(&question) > MAX_QUESTION_CHARS {
            question = prefix_chars(&question, MAX_QUESTION_CHARS - 1) + "…";
        }

        QuestionVerdict { is_question: true, confidence: top.confidence, reason: top.reasons.join("+"), question }
    }
}

/// Okno transkryptu w prompcie. Było 6 wypowiedzi / 1200 znaków i przy
/// „masz jakieś pytania?" model dopytywał o to, co padło 15 minut wcześniej.
/// 4000 znaków to ~1000 tokenów: zmierzone 2026-10-06 na Sonnecie: TTFT 0,9 s
/// -> 2,0 s w medianie, ale pytania do zadania przestały powtarzać to, co
/// rekruter już powiedział.
pub const DEFAULT_CONTEXT_SEGMENTS: usize = 30;
const MAX_CONTEXT_CHARS: usize = 4000;
/// Twardy limit kontekstu projektu — dłuższy opis to wolniejsza odpowiedź.
const MAX_PROJECT_CONTEXT_CHARS: usize = 8000;

/// Buduje prompt: pytanie + minimalny konieczny kontekst.
///
/// Kontekst trzymamy krótko celowo — każdy dodatkowy token to opóźnienie,
/// a odpowiedź ma paść, zanim rozmowa pójdzie dalej.
/// Swift: domyślnie `segments = []`, `title = ""`, `project_context = ""`,
/// `max_segments = DEFAULT_CONTEXT_SEGMENTS`.
pub fn build_prompt(
    question: &str,
    segments: &[Segment],
    title: &str,
    project_context: &str,
    max_segments: usize,
) -> Option<String> {
    let asked = Text::normalize(question);
    if asked.is_empty() {
        return None;
    }

    let recent: Vec<String> = segments[segments.len().saturating_sub(max_segments)..]
        .iter()
        .map(|s| format!("{}: {}", s.speaker, Text::normalize(&s.text)))
        .filter(|l| char_len(l) > 3)
        .collect();

    let mut context = recent.join("\n");
    if char_len(&context) > MAX_CONTEXT_CHARS {
        context = "…".to_string() + &suffix_chars(&context, MAX_CONTEXT_CHARS);
    }

    let mut parts: Vec<String> = Vec::new();
    let t = Text::normalize(title);
    if !t.is_empty() {
        parts.push(format!("Spotkanie: {t}"));
    }
    // Kontekst projektu idzie przed transkryptem: to tło, na którym toczy się
    // rozmowa, a nie to, co przed chwilą padło.
    if !project_context.is_empty() {
        let trimmed = if char_len(project_context) > MAX_PROJECT_CONTEXT_CHARS {
            prefix_chars(project_context, MAX_PROJECT_CONTEXT_CHARS) + "…"
        } else {
            project_context.to_string()
        };
        parts.push(format!("Kontekst projektu, o którym jest rozmowa:\n{trimmed}"));
    }
    if !context.is_empty() {
        parts.push(format!("Ostatnie wypowiedzi:\n{context}"));
    }
    parts.push(format!("Pytanie, na które masz odpowiedzieć:\n{asked}"));
    Some(parts.join("\n\n"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AnswerStatus {
    Pending,
    Streaming,
    Done,
    Error,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantItem {
    pub id: String,
    pub question: String,
    pub speaker: Option<String>,
    pub at: f64,
    pub auto: bool,
    pub status: AnswerStatus,
    pub answer: String,
    pub error: Option<String>,
    pub duration_ms: Option<f64>,
    /// Czas do pierwszego tokenu — mierzony, bo to jedyna latencja, którą widać.
    pub ttft_ms: Option<f64>,
    /// Czy do pytania dołączono obrazek. Sam obrazek zostaje poza rdzeniem —
    /// tutaj wystarczy wiedzieć, że interfejs ma pokazać znacznik.
    #[serde(default)]
    pub has_image: bool,
}

/// Stan pytań i odpowiedzi w trakcie rozmowy.
#[derive(Debug, Clone)]
pub struct AssistantSession {
    items: HashMap<String, AssistantItem>,
    order: Vec<String>,
    sequence: u64,
    pub max_items: usize,
    revision: u64,
}

impl Default for AssistantSession {
    fn default() -> Self {
        Self::new(30)
    }
}

impl AssistantSession {
    pub fn new(max_items: usize) -> Self {
        Self { items: HashMap::new(), order: Vec::new(), sequence: 0, max_items, revision: 0 }
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Swift: `speaker = nil`, `at = nowMs()`, `auto = false`, `hasImage = false`.
    pub fn add(&mut self, question: &str, speaker: Option<&str>, at: f64, auto: bool, has_image: bool) -> AssistantItem {
        self.sequence += 1;
        let id = format!("q{}", self.sequence);
        let item = AssistantItem {
            id: id.clone(),
            question: Text::normalize(question),
            speaker: speaker.map(String::from),
            at,
            auto,
            status: AnswerStatus::Pending,
            answer: String::new(),
            error: None,
            duration_ms: None,
            ttft_ms: None,
            has_image,
        };
        self.items.insert(id.clone(), item.clone());
        self.order.push(id);
        self.trim();
        self.revision += 1;
        item
    }

    pub fn append(&mut self, id: &str, delta: &str, ttft_ms: Option<f64>) -> Option<AssistantItem> {
        let item = self.items.get_mut(id)?;
        if item.answer.is_empty() && item.ttft_ms.is_none() {
            if let Some(t) = ttft_ms {
                item.ttft_ms = Some(t);
            }
        }
        item.answer.push_str(delta);
        item.status = AnswerStatus::Streaming;
        let out = item.clone();
        self.revision += 1;
        Some(out)
    }

    pub fn complete(&mut self, id: &str, answer: Option<&str>, duration_ms: Option<f64>) -> Option<AssistantItem> {
        let item = self.items.get_mut(id)?;
        if let Some(a) = answer {
            item.answer = a.to_string();
        }
        item.status = AnswerStatus::Done;
        item.duration_ms = duration_ms;
        let out = item.clone();
        self.revision += 1;
        Some(out)
    }

    pub fn fail(&mut self, id: &str, error: &str) -> Option<AssistantItem> {
        let item = self.items.get_mut(id)?;
        item.status = AnswerStatus::Error;
        item.error = Some(error.to_string());
        let out = item.clone();
        self.revision += 1;
        Some(out)
    }

    pub fn get(&self, id: &str) -> Option<&AssistantItem> {
        self.items.get(id)
    }

    pub fn all(&self) -> Vec<AssistantItem> {
        self.order.iter().filter_map(|id| self.items.get(id).cloned()).collect()
    }

    fn trim(&mut self) {
        while self.order.len() > self.max_items {
            let id = self.order.remove(0);
            self.items.remove(&id);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Found {
    pub segment_id: String,
    pub question: String,
    pub utterance: String,
    pub speaker: String,
    pub at: f64,
    pub confidence: f64,
    pub reason: String,
}

/// Wyławia pytania z napływających segmentów transkryptu.
///
/// Sprawdza wyłącznie segmenty domknięte: wypowiedź w trakcie jeszcze się
/// zmienia, a zadanie pytania w połowie zdania kosztuje czas i daje odpowiedź
/// na coś, co nie padło.
pub struct QuestionWatcher {
    seen: HashSet<String>,
    seen_order: Vec<String>,
    pub min_confidence: f64,
    pub max_seen: usize,
    pub on_question: Box<dyn FnMut(&Found) + Send>,
}

impl Default for QuestionWatcher {
    fn default() -> Self {
        Self::new(0.35, 500, Box::new(|_| {}))
    }
}

impl QuestionWatcher {
    pub fn new(min_confidence: f64, max_seen: usize, on_question: Box<dyn FnMut(&Found) + Send>) -> Self {
        Self { seen: HashSet::new(), seen_order: Vec::new(), min_confidence, max_seen, on_question }
    }

    pub fn scan(&mut self, segments: &[Segment]) -> Vec<Found> {
        let mut found = Vec::new();
        for segment in segments {
            if !segment.is_final || self.seen.contains(&segment.id) {
                continue;
            }
            self.seen.insert(segment.id.clone());
            self.seen_order.push(segment.id.clone());

            let verdict = QuestionDetector::detect(&segment.text);
            if !verdict.is_question || verdict.confidence < self.min_confidence {
                continue;
            }

            let item = Found {
                segment_id: segment.id.clone(),
                // Właściwe pytanie, bez dygresji przed nim — mniej tokenów, szybsza odpowiedź.
                question: if verdict.question.is_empty() { segment.text.clone() } else { verdict.question },
                utterance: segment.text.clone(),
                speaker: segment.speaker.clone(),
                at: segment.started_at,
                confidence: verdict.confidence,
                reason: verdict.reason,
            };
            (self.on_question)(&item);
            found.push(item);
        }
        self.trim();
        found
    }

    fn trim(&mut self) {
        while self.seen_order.len() > self.max_seen {
            let id = self.seen_order.remove(0);
            self.seen.remove(&id);
        }
    }

    pub fn reset(&mut self) {
        self.seen.clear();
        self.seen_order.clear();
    }
}
