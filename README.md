# call-whisper — Windows

Transkrypcja rozmów na żywo (whisper.cpp, lokalnie) z podpowiedziami. Port
aplikacji macOS z [vibebrosi-main/whisper](https://github.com/vibebrosi-main/whisper).

## Instalacja

Pobierz `call-whisper_*_x64-setup.exe` z [Releases](../../releases/latest)
i uruchom. Instalator wiezie silnik mowy (Vulkan na każde GPU, zapasowo CPU)
i ffmpeg. Model mowy (~490 MB) pobiera się przy pierwszym „Słuchaj".

Instalator nie jest podpisany — SmartScreen pokaże ostrzeżenie:
„Więcej informacji" → „Uruchom mimo to".

## Aktualizacje

Każdy push na `main` buduje nowe wydanie (GitHub Actions). Aplikacja co 30 min
sprawdza najnowsze i instaluje je po zakończonej rozmowie. Tylko pobiera.

## Budowanie lokalnie (Windows)

Rust, Visual Studio Build Tools (C++), Vulkan SDK, CMake. Silnik mowy i ffmpeg
do `vendor/` — kroki jak w `.github/workflows/build.yml`. Potem:

```
cargo install tauri-cli --version "^2"
cargo tauri build
```

Testy rdzenia: `cargo test -p cw-core`.

## Stan portu

- [x] Etap 1: dźwięk komputera (WASAPI loopback) i mikrofon z wyborem
  urządzenia, whisper na żywo, transkrypt, kopiowanie, eksport, ustawienia,
  gotowość, aktualizacje
- [ ] Etap 2: podpowiedzi (Claude Code, API), wykrywanie pytań, zrzuty ekranu
- [ ] Etap 3: nakładka, pasek u góry ekranu, import nagrań
- [ ] Etap 4: rozpoznawanie głosów, wykrywanie rozmów, OBS
