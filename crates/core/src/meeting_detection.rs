//! Rozstrzyganie, czy właśnie trwa rozmowa — pomysł z OpenWhispr
//! (`meetingDetectionEngine`), uproszczony do tego, co da się sprawdzić
//! bez uprawnień: czy ktoś trzyma mikrofon i czy działa znany komunikator.
//!
//! Sam mikrofon nie wystarcza (dyktowanie, notatka głosowa), sama aplikacja
//! też nie (Slack i Teams działają cały dzień). Dopiero oba naraz znaczą
//! rozmowę z dużym prawdopodobieństwem.
//!
//! Port z `MeetingDetection.swift`. Identyfikatory to bundle ID z macOS —
//! warstwa platformy na Windows musi podać swoje albo je zmapować.

use std::collections::HashSet;

pub struct MeetingDetection;

/// (bundle ID, nazwa)
pub const KNOWN_APPS: &[(&str, &str)] = &[
    ("us.zoom.xos", "Zoom"),
    ("com.microsoft.teams2", "Microsoft Teams"),
    ("com.microsoft.teams", "Microsoft Teams"),
    ("com.apple.FaceTime", "FaceTime"),
    ("com.tinyspeck.slackmacgap", "Slack"),
    ("com.hnc.Discord", "Discord"),
    ("com.cisco.webexmeetingsapp", "Webex"),
    ("Cisco-Systems.Spark", "Webex"),
    ("net.whatsapp.WhatsApp", "WhatsApp"),
    ("ru.keepcoder.Telegram", "Telegram"),
    ("com.skype.skype", "Skype"),
];

/// Przeglądarki: Google Meet nie ma aplikacji, więc mikrofon trzymany przez
/// przeglądarkę też liczymy — z nazwą ogólną, bo karty nie widać.
pub const BROWSERS: &[(&str, &str)] = &[
    ("com.google.Chrome", "przeglądarce"),
    ("com.apple.Safari", "przeglądarce"),
    ("company.thebrowser.Browser", "przeglądarce"),
    ("org.mozilla.firefox", "przeglądarce"),
    ("com.microsoft.edgemac", "przeglądarce"),
    ("com.brave.Browser", "przeglądarce"),
];

impl MeetingDetection {
    /// Nazwa aplikacji rozmowy albo `None`.
    ///
    /// - `mic_in_use`: czy jakikolwiek proces trzyma wejście audio
    /// - `running`: identyfikatory uruchomionych aplikacji
    /// - `frontmost`: aplikacja na pierwszym planie — przy kilku kandydatach
    ///   ona wygrywa, bo to z nią użytkownik właśnie rozmawia
    pub fn active_meeting(mic_in_use: bool, running: &HashSet<String>, frontmost: Option<&str>) -> Option<&'static str> {
        if !mic_in_use {
            return None;
        }
        let candidates: Vec<&(&str, &str)> = KNOWN_APPS.iter().filter(|(id, _)| running.contains(*id)).collect();
        if let Some(front) = frontmost {
            if let Some(hit) = candidates.iter().find(|(id, _)| *id == front) {
                return Some(hit.1);
            }
        }
        if let Some(first) = candidates.first() {
            return Some(first.1);
        }
        if let Some(front) = frontmost {
            if let Some(b) = BROWSERS.iter().find(|(id, _)| *id == front) {
                return Some(b.1);
            }
        }
        None
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MeetingChange {
    Started(String),
    Ended(String),
}

/// Histereza: mikrofon potrafi mrugnąć na ułamek sekundy (podgląd
/// w ustawieniach, dźwięk powiadomienia). Rozmowa zaczyna się po `on_after`
/// kolejnych trafieniach i kończy po `off_after` pudłach.
#[derive(Debug, Clone)]
pub struct Debouncer {
    pub on_after: usize,
    pub off_after: usize,
    active: Option<String>,
    hits: usize,
    misses: usize,
}

impl Default for Debouncer {
    fn default() -> Self {
        Self::new(2, 5)
    }
}

impl Debouncer {
    pub fn new(on_after: usize, off_after: usize) -> Self {
        Self { on_after, off_after, active: None, hits: 0, misses: 0 }
    }

    pub fn active(&self) -> Option<&str> {
        self.active.as_deref()
    }

    pub fn feed(&mut self, meeting: Option<&str>) -> Option<MeetingChange> {
        if let Some(meeting) = meeting {
            self.misses = 0;
            self.hits += 1;
            if self.active.is_none() && self.hits >= self.on_after {
                self.active = Some(meeting.to_string());
                return Some(MeetingChange::Started(meeting.to_string()));
            }
        } else {
            self.hits = 0;
            self.misses += 1;
            if self.misses >= self.off_after {
                if let Some(current) = self.active.take() {
                    return Some(MeetingChange::Ended(current));
                }
            }
        }
        None
    }
}
