//! Czysta część współpracy z OBS: konfiguracja obs-websocket, uwierzytelnienie
//! protokołu v5 i nazwa pliku z transkryptem. Gniazdo siedzi w warstwie platformy.
//!
//! Protokół: https://github.com/obsproject/obs-websocket/blob/master/docs/generated/protocol.md
//! Port z `OBS.swift`. SHA-256 i base64 liczymy sami (Swift brał CryptoKit),
//! żeby nie dokładać zależności dla dwóch wywołań.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;

pub struct Obs;

/// Ustawienia serwera z pliku, który OBS zapisuje sam. Czytamy je zamiast
/// kazać przepisywać hasło do call-whisper: oba programy działają na tym
/// samym koncie, więc plik i tak jest w zasięgu.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServerConfig {
    pub enabled: bool,
    pub port: i64,
    pub password: Option<String>,
}

impl ServerConfig {
    pub fn parse(data: &[u8]) -> Option<ServerConfig> {
        let json: Value = serde_json::from_slice(data).ok()?;
        let obj = json.as_object()?;
        let auth_required = obj.get("auth_required").and_then(Value::as_bool).unwrap_or(true);
        let password = obj.get("server_password").and_then(Value::as_str);
        Some(ServerConfig {
            enabled: obj.get("server_enabled").and_then(Value::as_bool).unwrap_or(false),
            port: obj.get("server_port").and_then(Value::as_i64).unwrap_or(4455),
            password: if auth_required && !password.unwrap_or("").is_empty() { password.map(String::from) } else { None },
        })
    }

    /// Plik konfiguracji obs-websocket. Swift: `~/Library/Application Support/obs-studio/…`;
    /// na Windows OBS trzyma go w `%APPDATA%\obs-studio\…`.
    pub fn default_path() -> Option<PathBuf> {
        let base = if cfg!(windows) {
            PathBuf::from(std::env::var_os("APPDATA")?)
        } else if cfg!(target_os = "macos") {
            PathBuf::from(std::env::var_os("HOME")?).join("Library/Application Support")
        } else {
            std::env::var_os("XDG_CONFIG_HOME")
                .map(PathBuf::from)
                .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?
        };
        Some(base.join("obs-studio").join("plugin_config").join("obs-websocket").join("config.json"))
    }

    pub fn load(path: &Path) -> Option<ServerConfig> {
        std::fs::read(path).ok().and_then(|d| Self::parse(&d))
    }

    /// `server_enabled: true` w istniejącym pliku, reszta bez zmian.
    /// Wolno to robić tylko przy zamkniętym OBS - działający przeczytał
    /// plik przy starcie i nadpisze go przy wyjściu.
    ///
    /// Pliku, którego nie ma, celowo nie zakładamy: OBS przy pierwszym
    /// starcie sam generuje hasło, a serwer bez hasła słucha na wszystkich
    /// interfejsach, czyli dałby sterowanie OBS-em całej sieci lokalnej.
    pub fn enable_server(path: &Path) -> bool {
        let Ok(data) = std::fs::read(path) else { return false };
        let Ok(Value::Object(mut json)) = serde_json::from_slice::<Value>(&data) else { return false };
        if json.get("server_enabled").and_then(Value::as_bool) == Some(true) {
            return true;
        }
        json.insert("server_enabled".into(), Value::Bool(true));
        // serde_json::Map bez `preserve_order` jest posortowana — jak `.sortedKeys`.
        let Ok(out) = serde_json::to_vec_pretty(&Value::Object(json)) else { return false };
        // Zapis atomowy: plik tymczasowy obok i podmiana.
        let tmp = path.with_extension("json.cw-tmp");
        if std::fs::write(&tmp, out).is_err() {
            return false;
        }
        if std::fs::rename(&tmp, path).is_err() {
            let _ = std::fs::remove_file(&tmp);
            return false;
        }
        true
    }
}

/// Stan nagrywania z `RecordStateChanged`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecordEvent {
    Started { path: Option<String> },
    Stopped { path: Option<String> },
}

impl Obs {
    /// Bit subskrypcji zdarzeń wyjść (nagrywanie, stream, replay buffer).
    pub const OUTPUTS_EVENT_SUBSCRIPTION: i64 = 1 << 6;

    /// `authentication` z wiadomości Identify:
    /// base64(sha256(base64(sha256(hasło + sól)) + wyzwanie)).
    pub fn authentication(password: &str, salt: &str, challenge: &str) -> String {
        let secret = base64(&sha256(format!("{password}{salt}").as_bytes()));
        base64(&sha256(format!("{secret}{challenge}").as_bytes()))
    }

    /// Transkrypt ląduje obok nagrania, z tą samą nazwą: `rozmowa.mkv`
    /// -> `rozmowa.md`. Dzięki temu pliki trzymają się razem w eksploratorze.
    pub fn transcript_path(recording: &str) -> PathBuf {
        Path::new(recording).with_extension("md")
    }

    /// Wyłuskuje start i stop nagrywania ze zdarzenia (op 5). Pauzę, wznowienie
    /// i stany przejściowe („starting", „stopping") pomijamy.
    pub fn record_event(message: &Value) -> Option<RecordEvent> {
        if message.get("op").and_then(Value::as_i64) != Some(5) {
            return None;
        }
        let d = message.get("d")?.as_object()?;
        if d.get("eventType").and_then(Value::as_str) != Some("RecordStateChanged") {
            return None;
        }
        let data = d.get("eventData")?.as_object()?;
        let state = data.get("outputState")?.as_str()?;
        let path = data.get("outputPath").and_then(Value::as_str).filter(|p| !p.is_empty()).map(String::from);
        match state {
            "OBS_WEBSOCKET_OUTPUT_STARTED" => Some(RecordEvent::Started { path }),
            "OBS_WEBSOCKET_OUTPUT_STOPPED" => Some(RecordEvent::Stopped { path }),
            _ => None,
        }
    }
}

fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let b = [chunk[0], *chunk.get(1).unwrap_or(&0), *chunk.get(2).unwrap_or(&0)];
        let n = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
        out.push(ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(ALPHABET[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 { ALPHABET[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if chunk.len() > 2 { ALPHABET[n as usize & 63] as char } else { '=' });
    }
    out
}

fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];
    let mut msg = data.to_vec();
    let bit_len = (data.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for block in msg.chunks(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([block[4 * i], block[4 * i + 1], block[4 * i + 2], block[4 * i + 3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16].wrapping_add(s0).wrapping_add(w[i - 7]).wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ (!e & g);
            let t1 = hh.wrapping_add(s1).wrapping_add(ch).wrapping_add(K[i]).wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        for (x, y) in h.iter_mut().zip([a, b, c, d, e, f, g, hh]) {
            *x = x.wrapping_add(y);
        }
    }
    let mut out = [0u8; 32];
    for (i, v) in h.iter().enumerate() {
        out[4 * i..4 * i + 4].copy_from_slice(&v.to_be_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_i_base64_wektory_wzorcowe() {
        let hex = |b: &[u8]| b.iter().map(|x| format!("{x:02x}")).collect::<String>();
        assert_eq!(hex(&sha256(b"")), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
        assert_eq!(hex(&sha256(b"abc")), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
    }
}
