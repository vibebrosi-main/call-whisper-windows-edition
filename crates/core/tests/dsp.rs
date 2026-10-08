//! DSP i diaryzacja na syntetycznych głosach — te same co w testach JS.
mod common;

use std::f64::consts::PI;

use common::synth::{self, ANNA, FRAME, HOP, JAN, SR};
use common::*;
use cw_core::*;

/// FFT musi zgadzać się z naiwną DFT — to jedyny sposób, żeby upewnić się,
/// że pakowanie wyniku i skalowanie rozpakowaliśmy dobrze.
#[test]
fn fft_zgadza_sie_z_naiwna_dft() {
    let n = 64;
    let signal: Vec<f32> = (0..n)
        .map(|i| {
            ((2.0 * PI * 5.0 * i as f64 / n as f64).sin() + 0.5 * (2.0 * PI * 13.0 * i as f64 / n as f64).cos() + 0.1)
                as f32
        })
        .collect();

    let fft = FftProcessor::new(n).unwrap();
    let fast = fft.power_spectrum(&signal);

    for (k, &got) in fast.iter().enumerate().take(n / 2 + 1) {
        let (mut re, mut im) = (0.0, 0.0);
        for (i, &x) in signal.iter().enumerate() {
            let angle = 2.0 * PI * k as f64 * i as f64 / n as f64;
            re += x as f64 * angle.cos();
            im -= x as f64 * angle.sin();
        }
        let expected = re * re + im * im;
        assert!((got as f64 - expected).abs() < 1e-2, "prążek {k}: {got} vs {expected}");
    }
}

#[test]
fn fft_odrzuca_rozmiar_nie_bedacy_potega_dwojki() {
    assert!(FftProcessor::new(100).is_none());
    assert!(FftProcessor::new(512).is_some());
}

#[test]
fn l2_normalize_i_cosinus() {
    let v = l2_normalize(&[3.0, 4.0]);
    assert!(((v[0] * v[0] + v[1] * v[1]).sqrt() - 1.0).abs() < 1e-12);
    assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-12);

    let orthogonal = l2_normalize(&[-4.0, 3.0]);
    assert!(cosine_similarity(&v, &orthogonal).abs() < 1e-12);

    let expected = f64s(fx("norm34"));
    for (a, e) in v.iter().zip(&expected) {
        assert!((a - e).abs() < 1e-12);
    }
}

/// Swift liczył MFCC w Float (Accelerate), JS w Double — drobna rozbieżność
/// jest oczekiwana, ale barwa głosu musi zostać ta sama.
#[test]
fn mfcc_zgadza_sie_z_js() {
    let extractor = MfccExtractor::default();
    let signal = synth::voice(&ANNA, 0.2, 1.0, 7.0);
    let expected: Vec<Vec<f64>> = fx("mfccFrames").as_array().unwrap().iter().map(f64s).collect();

    for (i, exp) in expected.iter().enumerate().take(3) {
        let mfcc = extractor.frame_to_mfcc(&synth::frame_at(&signal, i * HOP));
        assert_eq!(mfcc.len(), exp.len());
        for j in 0..mfcc.len() {
            assert!((mfcc[j] - exp[j]).abs() < 0.05, "ramka {i}, współczynnik {j}: {} vs {}", mfcc[j], exp[j]);
        }
    }
}

#[test]
fn energia_ramki_zgadza_sie_z_js() {
    let signal = synth::voice(&ANNA, 0.2, 1.0, 7.0);
    let expected = f64s(fx("energyDb"));
    for (i, exp) in expected.iter().enumerate().take(3) {
        let db = MfccExtractor::frame_energy_db(&synth::frame_at(&signal, i * HOP));
        assert!((db - exp).abs() < 1e-3, "ramka {i}: {db} vs {exp}");
    }
}

/// VAD musi otworzyć i zamknąć wypowiedź w tych samych ramkach co JS.
/// To pilnuje statystyki minimum — regresja tutaj oznacza, że wentylator
/// znowu jest mową albo że długa wypowiedź się ucina.
#[test]
fn vad_otwiera_i_zamyka_w_tych_samych_ramkach() {
    let mut signal = synth::silence(0.4, 3.0);
    signal.extend(synth::voice(&ANNA, 0.8, 1.0, 7.0));
    signal.extend(synth::silence(0.6, 3.0));

    let mut vad = Vad::default();
    let mut events: Vec<(&str, usize)> = Vec::new();
    let mut start = 0;
    while start + FRAME <= signal.len() {
        let st = vad.push(MfccExtractor::frame_energy_db(&synth::frame_at(&signal, start)));
        if st.started {
            events.push(("start", start / HOP));
        }
        if st.ended {
            events.push(("end", start / HOP));
        }
        start += HOP;
    }

    let expected = cases("vadEvents");
    assert_eq!(events.len(), expected.len(), "liczba zdarzeń VAD: {events:?}");
    for (a, e) in events.iter().zip(expected) {
        assert_eq!(a.0, s(&e["event"]));
        assert_eq!(a.1 as u64, e["frame"].as_u64().unwrap(), "ramka zdarzenia {}", a.0);
    }
}

/// Pełny łańcuch diaryzacji na dwóch syntetycznych głosach.
/// Anna -> Jan -> Anna musi dać dwóch mówców i etykiety 0,1,0.
#[test]
fn diaryzacja_rozdziela_dwa_glosy() {
    let mut signal = synth::silence(0.3, 3.0);
    signal.extend(synth::voice(&ANNA, 1.2, 1.0, 11.0));
    signal.extend(synth::silence(0.5, 3.0));
    signal.extend(synth::voice(&JAN, 1.2, 1.0, 13.0));
    signal.extend(synth::silence(0.5, 3.0));
    signal.extend(synth::voice(&ANNA, 1.2, 1.0, 17.0));
    signal.extend(synth::silence(0.4, 3.0));

    let mut diarizer = Diarizer::default();
    let mut start = 0;
    while start + FRAME <= signal.len() {
        diarizer.push_frame(&synth::frame_at(&signal, start), start as f64 / SR * 1000.0);
        start += HOP;
    }
    diarizer.flush(signal.len() as f64 / SR * 1000.0);

    let expected_turns = cases("turns");
    let expected_count = fx("speakerCount").as_u64().unwrap() as usize;

    assert_eq!(diarizer.speaker_count(), expected_count, "liczba mówców");
    assert_eq!(diarizer.turns().len(), expected_turns.len(), "liczba tur");

    let labels: Vec<usize> = diarizer.turns().iter().map(|t| t.speaker).collect();
    let expected_labels: Vec<usize> = expected_turns.iter().map(|t| t["speaker"].as_u64().unwrap() as usize).collect();
    assert_eq!(labels, expected_labels, "etykiety mówców");

    // Ta sama osoba na początku i na końcu — to jest cały sens diaryzacji.
    assert_eq!(labels.first(), labels.last());
    assert!(labels.len() >= 3 && labels[0] != labels[1]);

    // Granice tur mogą przesunąć się o ramkę przez arytmetykę, ale nie o więcej.
    for (a, e) in diarizer.turns().iter().zip(expected_turns) {
        let exp_start = e["startMs"].as_f64().unwrap();
        let exp_end = e["endMs"].as_f64().unwrap();
        assert!((a.start_ms - exp_start).abs() <= 20.0, "start tury: {} vs {exp_start}", a.start_ms);
        assert!((a.end_ms - exp_end).abs() <= 20.0, "koniec tury: {} vs {exp_end}", a.end_ms);
    }
}

#[test]
fn framer_tnie_na_ramki_ze_skokiem() {
    let mut framer = Framer::new(4, 2);
    let first = framer.push(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    assert_eq!(first.len(), 1);
    assert_eq!(first[0].frame, vec![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(first[0].start_sample, 0);

    // Bufor trzyma [3,4,5]; po dołożeniu 6,7 wychodzi dokładnie jedna
    // kolejna ramka, przesunięta o skok — nie dwie.
    let second = framer.push(&[6.0, 7.0]);
    assert_eq!(second.len(), 1);
    assert_eq!(second[0].frame, vec![3.0, 4.0, 5.0, 6.0]);
    assert_eq!(second[0].start_sample, 2);

    let third = framer.push(&[8.0, 9.0]);
    assert_eq!(third.len(), 1);
    assert_eq!(third[0].frame, vec![5.0, 6.0, 7.0, 8.0]);
    assert_eq!(third[0].start_sample, 4);
}

#[test]
fn tracker_zaklada_nowego_mowce_gdy_glos_odlegly() {
    let mut tracker = SpeakerTracker::new(0.9, 8, 0.85);
    let a = l2_normalize(&[1.0, 0.0, 0.0, 0.0]);
    let b = l2_normalize(&[0.0, 1.0, 0.0, 0.0]);
    assert!(tracker.assign(&a).is_new);
    assert!(tracker.assign(&b).is_new);
    assert!(!tracker.assign(&a).is_new);
    assert_eq!(tracker.count(), 2);
}

#[test]
fn tracker_nie_przekracza_limitu_mowcow() {
    let mut tracker = SpeakerTracker::new(0.99, 2, 0.85);
    tracker.assign(&l2_normalize(&[1.0, 0.0, 0.0]));
    tracker.assign(&l2_normalize(&[0.0, 1.0, 0.0]));
    let third = tracker.assign(&l2_normalize(&[0.0, 0.0, 1.0]));
    assert!(!third.is_new);
    assert_eq!(tracker.count(), 2);
}
