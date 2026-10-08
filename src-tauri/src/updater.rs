//! Aktualizacja z GitHuba: CI buduje instalator po każdym pushu na main
//! i publikuje go z `latest.json`. Tu tylko pobieramy — nigdy nic nie
//! wysyłamy. Paczka jest podpisana kluczem updatera, więc podmieniony plik
//! z sieci nie przejdzie weryfikacji.

use crate::recorder::Recorder;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Manager};
use tauri_plugin_updater::UpdaterExt;

const INTERVAL: Duration = Duration::from_secs(30 * 60);
/// Co ile sprawdzamy, czy nasłuch już się skończył.
const IDLE_POLL: Duration = Duration::from_secs(60);

pub fn spawn(app: AppHandle) {
    tauri::async_runtime::spawn(async move {
        // Chwila po starcie: najpierw okno, potem sieć.
        tokio::time::sleep(Duration::from_secs(10)).await;
        let mut last = Instant::now() - INTERVAL;
        loop {
            let waiting = recorder(&app).state().update.is_some();
            if waiting || last.elapsed() >= INTERVAL {
                last = Instant::now();
                check(&app, false).await;
            }
            tokio::time::sleep(IDLE_POLL).await;
        }
    });
}

fn recorder(app: &AppHandle) -> Arc<Recorder> {
    app.state::<Arc<Recorder>>().inner().clone()
}

/// `force` = kliknięte „Sprawdź teraz": działa też przy wyłączonej automatyce.
pub async fn check(app: &AppHandle, force: bool) {
    let rec = recorder(app);
    if !force && !rec.settings().auto_update {
        return;
    }
    let update = match app.updater() {
        Ok(updater) => updater.check().await,
        Err(err) => Err(err),
    };
    let update = match update {
        Ok(Some(update)) => update,
        Ok(None) => {
            rec.set_update(None);
            return;
        }
        Err(err) => {
            if force {
                rec.set_update(Some(format!("Nie udało się sprawdzić aktualizacji: {err}")));
            }
            return;
        }
    };
    // Nie przerywamy rozmowy: instalacja restartuje aplikację.
    if rec.is_busy() {
        rec.set_update(Some(format!("Nowa wersja {} — zainstaluję po nasłuchu", update.version)));
        return;
    }
    rec.set_update(Some(format!("Pobieram wersję {}…", update.version)));
    match update.download_and_install(|_, _| {}, || {}).await {
        Ok(()) => app.restart(),
        Err(err) => rec.set_update(Some(format!("Aktualizacja nie wyszła: {err}"))),
    }
}
