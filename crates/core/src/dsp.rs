//! DSP: FFT, bank filtrów mel, MFCC i VAD. Port z `DSP.swift`.

use std::f64::consts::PI;

/// FFT rzeczywista, radix-2.
///
/// Swift liczył ją na Accelerate (vDSP, Float). Tu własna implementacja
/// iteracyjna w `f64` — bez zależności, a wynik oddajemy jako `f32` jak oryginał.
#[derive(Debug, Clone)]
pub struct FftProcessor {
    pub size: usize,
    cos: Vec<f64>,
    sin: Vec<f64>,
    bitrev: Vec<usize>,
}

impl FftProcessor {
    /// `None`, gdy rozmiar nie jest potęgą dwójki większą od 1.
    pub fn new(size: usize) -> Option<Self> {
        if size <= 1 || size & (size - 1) != 0 {
            return None;
        }
        let bits = size.trailing_zeros();
        let bitrev = (0..size).map(|i| i.reverse_bits() >> (usize::BITS - bits)).collect();
        let half = size / 2;
        let cos = (0..half).map(|k| (2.0 * PI * k as f64 / size as f64).cos()).collect();
        let sin = (0..half).map(|k| (2.0 * PI * k as f64 / size as f64).sin()).collect();
        Some(Self { size, cos, sin, bitrev })
    }

    /// Widmo mocy sygnału rzeczywistego: |X(k)|² dla k = 0…n/2.
    pub fn power_spectrum(&self, frame: &[f32]) -> Vec<f32> {
        assert!(frame.len() == self.size, "FftProcessor: ramka musi mieć {} próbek", self.size);
        let n = self.size;
        let mut re = vec![0.0f64; n];
        let mut im = vec![0.0f64; n];
        for i in 0..n {
            re[self.bitrev[i]] = frame[i] as f64;
        }
        let mut len = 2;
        while len <= n {
            let step = n / len;
            for start in (0..n).step_by(len) {
                for j in 0..len / 2 {
                    let (wr, wi) = (self.cos[j * step], -self.sin[j * step]);
                    let a = start + j;
                    let b = a + len / 2;
                    let tr = re[b] * wr - im[b] * wi;
                    let ti = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
            len <<= 1;
        }
        (0..=n / 2).map(|k| (re[k] * re[k] + im[k] * im[k]) as f32).collect()
    }
}

pub fn next_power_of_two(n: usize) -> usize {
    let mut size = 1;
    while size < n {
        size <<= 1;
    }
    size
}

pub fn hz_to_mel(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

pub fn mel_to_hz(mel: f64) -> f64 {
    700.0 * (10f64.powf(mel / 2595.0) - 1.0)
}

/// Trójkątny filtr w skali mel — rzadki, trzyma tylko swój zakres prążków.
#[derive(Debug, Clone, PartialEq)]
pub struct MelFilter {
    pub start: usize,
    pub weights: Vec<f32>,
}

/// Bank filtrów mel. Swift: domyślnie 16 kHz, FFT 512, 26 filtrów, 80–7600 Hz.
pub fn mel_filterbank(sample_rate: f64, fft_size: usize, filters: usize, f_min: f64, f_max: f64) -> Vec<MelFilter> {
    let nyquist = sample_rate / 2.0;
    let top = f_max.min(nyquist);
    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(top);
    let bins = (fft_size / 2 + 1) as i64;

    // filters + 2 punktów: każdy filtr ma lewy, środkowy i prawy wierzchołek.
    let points: Vec<i64> = (0..filters + 2)
        .map(|i| {
            let mel = mel_min + (mel_max - mel_min) * i as f64 / (filters + 1) as f64;
            ((fft_size + 1) as f64 * mel_to_hz(mel) / sample_rate).floor() as i64
        })
        .collect();

    let mut bank = Vec::with_capacity(filters);
    for m in 1..=filters {
        let (left, center, right) = (points[m - 1], points[m], points[m + 1]);
        let start = left.max(0);
        let end = (bins - 1).min(right);
        let mut weights = vec![0f32; (end - start + 1).max(0) as usize];
        if end >= start {
            for k in start..=end {
                let mut value = 0.0;
                if k >= left && k <= center && center > left {
                    value = (k - left) as f64 / (center - left) as f64;
                } else if k > center && k <= right && right > center {
                    value = (right - k) as f64 / (right - center) as f64;
                }
                weights[(k - start) as usize] = value as f32;
            }
        }
        bank.push(MelFilter { start: start as usize, weights });
    }
    bank
}

/// Energie logarytmiczne w pasmach mel. Swift: domyślnie `floor = 1e-10`.
pub fn log_mel_energies(spectrum: &[f32], bank: &[MelFilter], floor: f64) -> Vec<f64> {
    bank.iter()
        .map(|filter| {
            let mut sum = 0.0;
            for (i, w) in filter.weights.iter().enumerate() {
                let bin = filter.start + i;
                if bin < spectrum.len() {
                    sum += spectrum[bin] as f64 * *w as f64;
                }
            }
            sum.max(floor).ln()
        })
        .collect()
}

/// DCT-II (ortonormalna) — pierwsze `count` współczynników.
pub fn dct2(input: &[f64], count: usize) -> Vec<f64> {
    let n = input.len();
    let scale0 = (1.0 / n as f64).sqrt();
    let scale = (2.0 / n as f64).sqrt();
    (0..count)
        .map(|k| {
            let mut sum = 0.0;
            for (i, x) in input.iter().enumerate() {
                sum += x * (PI * k as f64 * (2 * i + 1) as f64 / (2 * n) as f64).cos();
            }
            sum * if k == 0 { scale0 } else { scale }
        })
        .collect()
}

/// Ekstraktor MFCC — trzyma okno, bank filtrów i plan FFT.
/// Jedna instancja na źródło audio.
#[derive(Debug, Clone)]
pub struct MfccExtractor {
    pub sample_rate: f64,
    pub frame_size: usize,
    pub hop_size: usize,
    pub fft_size: usize,
    pub cepstra: usize,
    preemphasis_coeff: f32,
    window: Vec<f32>,
    bank: Vec<MelFilter>,
    fft: FftProcessor,
}

impl Default for MfccExtractor {
    fn default() -> Self {
        Self::new(16_000.0, 25.0, 10.0, 26, 12, 80.0, 7600.0, 0.97)
    }
}

impl MfccExtractor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        sample_rate: f64,
        frame_ms: f64,
        hop_ms: f64,
        mel_filters: usize,
        cepstra: usize,
        f_min: f64,
        f_max: f64,
        preemphasis: f64,
    ) -> Self {
        let frame_size = (frame_ms / 1000.0 * sample_rate).round() as usize;
        let hop_size = (hop_ms / 1000.0 * sample_rate).round() as usize;
        let fft_size = next_power_of_two(frame_size);
        // Okno Hamminga.
        let window = (0..frame_size)
            .map(|i| (0.54 - 0.46 * (2.0 * PI * i as f64 / (frame_size - 1) as f64).cos()) as f32)
            .collect();
        Self {
            sample_rate,
            frame_size,
            hop_size,
            fft_size,
            cepstra,
            preemphasis_coeff: preemphasis as f32,
            window,
            bank: mel_filterbank(sample_rate, fft_size, mel_filters, f_min, f_max),
            fft: FftProcessor::new(fft_size).expect("rozmiar FFT to potęga dwójki"),
        }
    }

    /// MFCC pojedynczej ramki (bez c0 — c0 to głośność, nie barwa głosu).
    pub fn frame_to_mfcc(&self, frame: &[f32]) -> Vec<f64> {
        assert!(frame.len() == self.frame_size, "MfccExtractor: ramka musi mieć {} próbek", self.frame_size);
        // Preemfaza y[n] = x[n] - a·x[n-1]: podbija wysokie częstotliwości,
        // gdzie siedzi informacja o formantach.
        let mut padded = vec![0f32; self.fft_size];
        padded[0] = frame[0] * self.window[0];
        for i in 1..self.frame_size {
            padded[i] = (frame[i] - self.preemphasis_coeff * frame[i - 1]) * self.window[i];
        }
        let spectrum = self.fft.power_spectrum(&padded);
        let mel = log_mel_energies(&spectrum, &self.bank, 1e-10);
        let cepstrum = dct2(&mel, self.cepstra + 1);
        cepstrum[1..].to_vec() // odrzucamy c0
    }

    /// Energia ramki w dB — używana przez VAD.
    pub fn frame_energy_db(frame: &[f32]) -> f64 {
        let mean_square = if frame.is_empty() {
            0.0
        } else {
            frame.iter().map(|&x| x as f64 * x as f64).sum::<f64>() / frame.len() as f64
        };
        10.0 * mean_square.max(1e-12).log10()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct VadState {
    pub speaking: bool,
    pub started: bool,
    pub ended: bool,
    pub loud: bool,
}

/// Detekcja aktywności głosowej (VAD) na energii ramki.
///
/// Podłogę szumu wyznaczamy metodą statystyki minimum: sygnał tniemy na
/// podokna i pamiętamy minimum z każdego z nich, a podłoga to minimum z całego
/// bufora. To odporne na dwa przeciwne przypadki, na których wykłada się
/// naiwna średnia ruchoma:
///
///  - stały hałas (wentylator, muzyka) — średnia uznaje go za mowę na zawsze,
///    minimum poprawnie ustawia się na jego poziomie;
///  - długa nieprzerwana wypowiedź — minimum trzyma się przerw międzysylabowych,
///    więc podłoga nie wspina się do poziomu mowy i nie ucina jej w połowie.
#[derive(Debug, Clone)]
pub struct Vad {
    /// O ile dB ponad podłogą szumu ramka liczy się jako mowa.
    pub threshold_db: f64,
    /// Ile kolejnych głośnych ramek otwiera wypowiedź (10 ms na ramkę).
    pub onset_frames: usize,
    /// Ile cichych ramek ją zamyka — krótkie pauzy w zdaniu nie mają jej ciąć.
    pub hangover_frames: usize,
    pub subwindow_frames: usize,
    pub initial_floor_db: f64,

    noise_floor_db: f64,
    speaking: bool,
    loud_run: usize,
    quiet_run: usize,
    sub_min: f64,
    sub_count: usize,
    ring: Vec<f64>,
    ring_index: usize,
}

impl Default for Vad {
    fn default() -> Self {
        Self::new(12.0, 3, 25, 50, 6, -70.0)
    }
}

impl Vad {
    pub fn new(
        threshold_db: f64,
        onset_frames: usize,
        hangover_frames: usize,
        subwindow_frames: usize,
        subwindows: usize,
        initial_floor_db: f64,
    ) -> Self {
        Self {
            threshold_db,
            onset_frames,
            hangover_frames,
            subwindow_frames,
            initial_floor_db,
            noise_floor_db: initial_floor_db,
            speaking: false,
            loud_run: 0,
            quiet_run: 0,
            sub_min: f64::INFINITY,
            sub_count: 0,
            // Bufor wypełniony podłogą startową: zanim uzbieramy historię, VAD ma
            // działać, a nie uznawać wszystkiego za ciszę.
            ring: vec![initial_floor_db; subwindows],
            ring_index: 0,
        }
    }

    pub fn noise_floor_db(&self) -> f64 {
        self.noise_floor_db
    }

    pub fn speaking(&self) -> bool {
        self.speaking
    }

    pub fn push(&mut self, energy_db: f64) -> VadState {
        self.track_noise_floor(energy_db);
        let loud = energy_db > self.noise_floor_db + self.threshold_db;

        let mut started = false;
        let mut ended = false;

        if loud {
            self.loud_run += 1;
            self.quiet_run = 0;
            if !self.speaking && self.loud_run >= self.onset_frames {
                self.speaking = true;
                started = true;
            }
        } else {
            self.quiet_run += 1;
            self.loud_run = 0;
            if self.speaking && self.quiet_run >= self.hangover_frames {
                self.speaking = false;
                ended = true;
            }
        }

        VadState { speaking: self.speaking, started, ended, loud }
    }

    fn track_noise_floor(&mut self, energy_db: f64) {
        if energy_db < self.sub_min {
            self.sub_min = energy_db;
        }
        self.sub_count += 1;
        if self.sub_count >= self.subwindow_frames {
            self.ring[self.ring_index] = self.sub_min;
            self.ring_index = (self.ring_index + 1) % self.ring.len();
            self.sub_min = f64::INFINITY;
            self.sub_count = 0;
        }
        let mut floor = self.ring.iter().copied().reduce(f64::min).unwrap_or(self.initial_floor_db);
        if self.sub_min < floor {
            floor = self.sub_min;
        }
        self.noise_floor_db = floor;
    }

    pub fn reset(&mut self) {
        self.speaking = false;
        self.loud_run = 0;
        self.quiet_run = 0;
        self.sub_min = f64::INFINITY;
        self.sub_count = 0;
        self.ring.iter_mut().for_each(|v| *v = self.initial_floor_db);
        self.noise_floor_db = self.initial_floor_db;
    }
}
