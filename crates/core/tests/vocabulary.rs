//! Słownictwo dla whispera wyciągane z kontekstu projektu.
use cw_core::*;

const CONTEXT: &str = "## Kandydat
Stack: Next.js, React, **Nuxt**, Python/Django, Playwright. Pracuje z Pythonem.
Testy w Vitest i Playwright, React Native w Acme. Talent Huby w Brazylii.
Ściąga: `defineModel`, `useFetch`. Projekt m.in. example.com, wymóg DPP, CV i UI.";

#[test]
fn wyciaga_nazwy_technologii() {
    let terms = Vocabulary::terms(CONTEXT);
    for expected in ["Next.js", "Nuxt", "Playwright", "Vitest", "React Native", "example.com", "DPP"] {
        assert!(terms.iter().any(|t| t == expected), "brak {expected} w {terms:?}");
    }
}

#[test]
fn pomija_kod_skroty_i_poczatek_zdania() {
    let terms = Vocabulary::terms(CONTEXT);
    for unwanted in ["defineModel", "useFetch", "m.in", "CV", "UI", "Stack", "Testy", "Ściąga", "Projekt"] {
        assert!(!terms.iter().any(|t| t == unwanted), "{unwanted} nie powinno trafić do {terms:?}");
    }
}

#[test]
fn odmiany_zostaja_raz() {
    let terms = Vocabulary::terms("Używam Figma. Projekty w Figmie i z Figmy, znowu w Figmie.");
    assert_eq!(terms.iter().filter(|t| t.to_lowercase().starts_with("figm")).count(), 1);
}

#[test]
fn polskie_odmiany_na_koncu() {
    let terms = Vocabulary::terms(CONTEXT);
    let pos = |w: &str| terms.iter().position(|t| t == w).unwrap();
    let brazil = pos("Brazylii");
    assert!(pos("Next.js") < brazil);
    assert!(pos("Playwright") < brazil);
}

#[test]
fn scalanie_nie_dubluje_i_pilnuje_limitu() {
    let merged = Vocabulary::merge("React, Next.js, Python.", CONTEXT, 80);
    assert!(merged.starts_with("React, Next.js, Python, "), "{merged}");
    assert!(merged.chars().count() <= 80);
    assert!(!merged.contains("Pythonem"));
    assert_eq!(merged.matches("Next.js").count(), 1);
    assert_eq!(Vocabulary::merge("", "", Vocabulary::MAX_CHARS), "");
}
