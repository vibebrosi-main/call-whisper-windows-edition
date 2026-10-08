// Bez okna konsoli w wersji wydanej.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

fn main() {
    call_whisper_lib::run()
}
