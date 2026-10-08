//! Kołowy bufor audio.
//!
//! Diaryzator mówi „tura mówcy trwała od 12 340 ms do 17 800 ms" dopiero po jej
//! zakończeniu — więc surowe audio musi gdzieś czekać, żeby dało się je wtedy
//! wyciąć i wysłać do transkrypcji. Bufor trzyma ostatnie N sekund i pozwala
//! czytać po czasie, nie po indeksie.
//!
//! Port z `AudioRing.swift` (a ten z `extension/src/adapters/audio/ring.js`).

#[derive(Debug, Clone)]
pub struct AudioRing {
    pub sample_rate: f64,
    pub epoch_ms: f64,
    pub capacity: usize,
    buffer: Vec<f32>,
    written: usize,
}

impl Default for AudioRing {
    fn default() -> Self {
        Self::new(16_000.0, 90.0, 0.0)
    }
}

impl AudioRing {
    pub fn new(sample_rate: f64, seconds: f64, epoch_ms: f64) -> Self {
        let capacity = ((sample_rate * seconds).round() as i64).max(1) as usize;
        Self { sample_rate, epoch_ms, capacity, buffer: vec![0.0; capacity], written: 0 }
    }

    pub fn write(&mut self, samples: &[f32]) {
        for (i, &sample) in samples.iter().enumerate() {
            self.buffer[(self.written + i) % self.capacity] = sample;
        }
        self.written += samples.len();
    }

    /// Najstarszy czas, który jeszcze pamiętamy.
    pub fn oldest_ms(&self) -> f64 {
        let oldest_sample = self.written.saturating_sub(self.capacity);
        self.epoch_ms + oldest_sample as f64 / self.sample_rate * 1000.0
    }

    pub fn newest_ms(&self) -> f64 {
        self.epoch_ms + self.written as f64 / self.sample_rate * 1000.0
    }

    pub fn written_samples(&self) -> usize {
        self.written
    }

    /// Wycinek audio dla zakresu czasu. Zakres jest przycinany do tego, co bufor
    /// jeszcze pamięta — lepiej oddać krótszą wypowiedź niż nic.
    /// Zwraca `None`, gdy zakres wypadł całkowicie poza bufor.
    pub fn read_range(&self, start_ms: f64, end_ms: f64) -> Option<Vec<f32>> {
        if end_ms <= start_ms {
            return None;
        }
        let to_sample = |ms: f64| -> i64 { (((ms - self.epoch_ms) / 1000.0) * self.sample_rate).round() as i64 };
        let written = self.written as i64;
        let oldest = (written - self.capacity as i64).max(0);
        let from = oldest.max(to_sample(start_ms));
        let to = written.min(to_sample(end_ms));
        if to <= from {
            return None;
        }
        Some((from..to).map(|i| self.buffer[(i as usize) % self.capacity]).collect())
    }

    pub fn reset(&mut self) {
        self.buffer.iter_mut().for_each(|v| *v = 0.0);
        self.written = 0;
    }
}
