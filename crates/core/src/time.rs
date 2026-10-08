//! Formatowanie czasu. Port z `Time.swift` (a ten z `extension/src/core/time.js`).

use chrono::{Datelike, Local, TimeZone, Timelike};

pub struct TimeFormat;

fn pad(n: i64) -> String {
    format!("{:02}", n.unsigned_abs())
}

/// Składowe czasu lokalnego; poza zakresem zera, jak `?? 0` w Swifcie.
fn components(ts: f64) -> (i64, i64, i64, i64, i64, i64) {
    match Local.timestamp_millis_opt(ts.floor() as i64).single() {
        Some(d) => (
            d.year() as i64,
            d.month() as i64,
            d.day() as i64,
            d.hour() as i64,
            d.minute() as i64,
            d.second() as i64,
        ),
        None => (0, 0, 0, 0, 0, 0),
    }
}

fn total_seconds(ms: f64) -> i64 {
    let seconds = if ms.is_finite() { (ms / 1000.0).round() } else { 0.0 };
    (seconds as i64).max(0)
}

impl TimeFormat {
    /// Offset od początku sesji jako HH:MM:SS (godziny nieograniczone).
    pub fn offset(ms: f64) -> String {
        let total = total_seconds(ms);
        format!("{}:{}:{}", pad(total / 3600), pad((total % 3600) / 60), pad(total % 60))
    }

    /// Czas trwania w formacie zwięzłym: 42m 11s / 1h 02m / 8s
    pub fn duration(ms: f64) -> String {
        let total = total_seconds(ms);
        let (h, m, s) = (total / 3600, (total % 3600) / 60, total % 60);
        if h > 0 {
            return format!("{h}h {}m", pad(m));
        }
        if m > 0 {
            return format!("{m}m {}s", pad(s));
        }
        format!("{s}s")
    }

    pub fn local_date(ts: f64) -> String {
        let (y, mo, d, ..) = components(ts);
        format!("{y}-{}-{}", pad(mo), pad(d))
    }

    pub fn local_time(ts: f64, with_seconds: bool) -> String {
        let (.., h, mi, s) = components(ts);
        let base = format!("{}:{}", pad(h), pad(mi));
        if with_seconds {
            format!("{base}:{}", pad(s))
        } else {
            base
        }
    }

    pub fn local_date_time(ts: f64, with_seconds: bool) -> String {
        format!("{} {}", Self::local_date(ts), Self::local_time(ts, with_seconds))
    }

    /// Stempel do nazwy pliku: 2026-08-23_1015
    pub fn filename_stamp(ts: f64) -> String {
        let (.., h, mi, _) = components(ts);
        format!("{}_{}{}", Self::local_date(ts), pad(h), pad(mi))
    }
}

/// Czas w milisekundach od epoki — ten sam „zegar", którym posługiwał się JS.
pub fn now_ms() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64() * 1000.0)
        .unwrap_or(0.0)
}
