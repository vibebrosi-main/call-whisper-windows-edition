//! Bufor kołowy i koder WAV — obie rzeczy stoją między diaryzatorem
//! a whisperem, więc cicha rozbieżność z wersją JS dałaby transkrypcję
//! przesuniętą w czasie albo szum zamiast mowy.
mod common;

use common::*;
use cw_core::*;

#[test]
fn bufor_kolowy_zgadza_sie_z_js() {
    let mut ring = AudioRing::new(1000.0, 1.0, 0.0);
    ring.write(&(0..600).map(|i| i as f32 / 1000.0).collect::<Vec<_>>());

    let expected_a = f64s(fx("ringA"));
    let a = ring.read_range(100.0, 300.0).expect("zakres A");
    assert_eq!(a.len(), expected_a.len());
    for (x, y) in a.iter().zip(&expected_a) {
        assert!((*x - *y as f32).abs() < 1e-6);
    }

    ring.write(&(0..600).map(|i| (600 + i) as f32 / 1000.0).collect::<Vec<_>>());

    // Bufor się zawinął — najstarsze próbki wypadły i zakres jest przycinany
    // do tego, co jeszcze pamiętamy. Lepiej krótsza wypowiedź niż żadna.
    let expected_b = f64s(fx("ringB"));
    let b = ring.read_range(0.0, 400.0).expect("zakres B");
    assert_eq!(b.len(), expected_b.len(), "po zawinięciu");
    for (x, y) in b.iter().zip(&expected_b) {
        assert!((*x - *y as f32).abs() < 1e-6);
    }

    let meta = fx("ringMeta");
    assert_eq!(ring.oldest_ms(), meta["oldestMs"].as_f64().unwrap());
    assert_eq!(ring.newest_ms(), meta["newestMs"].as_f64().unwrap());
    assert_eq!(ring.written_samples() as u64, meta["written"].as_u64().unwrap());
    assert_eq!(ring.read_range(2000.0, 2500.0).is_none(), meta["outOfRange"].as_bool().unwrap());
    assert_eq!(ring.read_range(300.0, 300.0).is_none(), meta["inverted"].as_bool().unwrap());
}

#[test]
fn wav_zgadza_sie_z_js_co_do_bajtu() {
    let samples = [0.0f32, 0.5, -0.5, 1.0, -1.0, 0.25];
    let data = Wav::encode(&samples, 16_000, 1);
    let expected: Vec<u8> = fx("wavBytes").as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u8).collect();
    assert_eq!(data.len(), expected.len(), "długość WAV");
    assert_eq!(data, expected, "bajty WAV muszą się zgadzać co do jednego");
}

#[test]
fn czyszczenie_tekstu_whispera_zgadza_sie_z_js() {
    let cs = cases("cleanCases");
    assert!(!cs.is_empty());
    for c in cs {
        let input = s(&c["input"]);
        assert_eq!(Text::clean_whisper(input), s(&c["result"]), "cleanWhisper({input:?})");
    }
}

#[test]
fn naglowek_wav_jest_poprawny() {
    let data = Wav::encode(&[0.0; 160], 16_000, 1);
    assert_eq!(data.len(), 44 + 320);
    assert_eq!(&data[0..4], b"RIFF");
    assert_eq!(&data[8..12], b"WAVE");
    assert_eq!(&data[36..40], b"data");
}
