//! Czysta logika call-whisper: tekst, transkrypt, markdown, DSP, wykrywanie pytań.
//! Bez UI i bez audio systemu — dzięki temu chodzi w testach bez uprawnień.
//!
//! Port 1:1 z modułu Swift `CallWhisperCore`; jeden moduł Rust na plik Swift.

pub mod assistant;
pub mod audio_ring;
pub mod diarizer;
pub mod dsp;
pub mod markdown;
pub mod meeting_detection;
pub mod obs;
pub mod speaker_turns;
pub mod text;
pub mod time;
pub mod transcript_store;
pub mod vocabulary;
pub mod wav;

pub use assistant::*;
pub use audio_ring::*;
pub use diarizer::*;
pub use dsp::*;
pub use markdown::*;
pub use meeting_detection::*;
pub use obs::*;
pub use speaker_turns::*;
pub use text::*;
pub use time::*;
pub use transcript_store::*;
pub use vocabulary::*;
pub use wav::*;
