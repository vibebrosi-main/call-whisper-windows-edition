//! Wspólne dla testów: wektory referencyjne z JS i syntetyczne głosy.
#![allow(dead_code)]

use std::f64::consts::PI;
use std::sync::OnceLock;

use serde_json::Value;

/// Wektory referencyjne pochodzą z działającego kodu JS
/// (`macos/tools/gen-fixtures.mjs`), a nie z ręcznie przepisanych asercji.
pub fn fixtures() -> &'static Value {
    static F: OnceLock<Value> = OnceLock::new();
    F.get_or_init(|| serde_json::from_str(include_str!("../fixtures.json")).expect("fixtures.json"))
}

pub fn fx(key: &str) -> &'static Value {
    &fixtures()[key]
}

pub fn cases(key: &str) -> &'static Vec<Value> {
    fx(key).as_array().expect("tablica przypadków")
}

pub fn f64s(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

pub fn s(v: &Value) -> &str {
    v.as_str().unwrap()
}

/// Syntetyczne głosy — dokładnie te same, co w testach JS.
///
/// Generator szumu odtwarza arytmetykę JS na `f64` (łącznie z utratą
/// precyzji przy mnożeniu powyżej 2^53), więc sygnał jest bit w bit ten sam.
pub mod synth {
    use super::PI;

    pub const SR: f64 = 16_000.0;
    pub const FRAME: usize = 400; // 25 ms
    pub const HOP: usize = 160; // 10 ms

    struct Noise {
        state: f64,
    }

    impl Noise {
        fn next(&mut self) -> f64 {
            self.state = (self.state * 1103515245.0 + 12345.0) % 2147483648.0;
            self.state / 2147483648.0 - 0.5
        }
    }

    pub struct Voice {
        pub f0: f64,
        pub formants: &'static [(f64, f64)],
    }

    pub const ANNA: Voice = Voice { f0: 205.0, formants: &[(520.0, 0.30), (2350.0, 0.26), (3100.0, 0.12)] };
    pub const JAN: Voice = Voice { f0: 105.0, formants: &[(330.0, 0.32), (1100.0, 0.24), (2400.0, 0.10)] };

    pub fn voice(v: &Voice, seconds: f64, gain: f64, seed: f64) -> Vec<f64> {
        let mut random = Noise { state: seed };
        let length = (seconds * SR).round() as usize;
        (0..length)
            .map(|i| {
                let t = i as f64 / SR;
                // Lekka modulacja amplitudy imituje sylaby.
                let envelope = 0.7 + 0.3 * (2.0 * PI * 4.0 * t).sin();
                let mut value = 0.4 * (2.0 * PI * v.f0 * t).sin();
                for &(freq, amp) in v.formants {
                    value += amp * (2.0 * PI * freq * t + freq).sin();
                }
                gain * envelope * value + 0.004 * random.next()
            })
            .collect()
    }

    pub fn silence(seconds: f64, seed: f64) -> Vec<f64> {
        let mut random = Noise { state: seed };
        (0..(seconds * SR).round() as usize).map(|_| 0.0015 * random.next()).collect()
    }

    pub fn frame_at(signal: &[f64], start: usize) -> Vec<f32> {
        signal[start..start + FRAME].iter().map(|&x| x as f32).collect()
    }
}
