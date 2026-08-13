"""Голосовое управление браузером для ребёнка: слушает микрофон, распознаёт речь
офлайн (Faster-Whisper) и управляет Chromium через Selenium.

Работает как переключатель: пока не сказано start_word ("запустись") - ассистент
ничего не делает. После start_word выполняет любые распознанные команды, пока не
услышит stop_word ("остановись") - тогда снова засыпает и ждёт start_word.
"""
import logging
import os
import signal
import sys
import time

import yaml

from audio import Recorder, Transcriber, play_beep
from browser import BrowserController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("main")

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")


def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    cfg["start_word"] = os.environ.get("START_WORD", cfg.get("start_word", "запустись")).strip().lower()
    cfg["stop_word"] = os.environ.get("STOP_WORD", cfg.get("stop_word", "остановись")).strip().lower()
    cfg["auto_stop_after_s"] = int(os.environ.get("AUTO_STOP_AFTER_S", cfg.get("auto_stop_after_s", 600)))
    cfg.setdefault("favorites", {})
    cfg.setdefault("strict_whitelist", False)
    cfg.setdefault("home_url", "https://www.youtubekids.com/")
    return cfg


def handle_command(browser: BrowserController, cfg: dict, command: str) -> None:
    if not command:
        return
    log.info("Команда: «%s»", command)

    favorites = cfg["favorites"]
    for phrase, url in favorites.items():
        if phrase in command:
            log.info("Открываю избранное: %s", phrase)
            browser.open_url(url)
            return

    if any(w in command for w in ("пауза", "стоп видео", "останови")):
        browser.toggle_play_pause()
    elif any(w in command for w in ("играй", "продолжи", "запусти видео")):
        browser.toggle_play_pause()
    elif any(w in command for w in ("погромче", "громче")):
        browser.volume_up()
    elif any(w in command for w in ("потише", "тише")):
        browser.volume_down()
    elif any(w in command for w in ("весь экран", "полный экран", "во весь экран")):
        browser.fullscreen()
    elif "назад" in command:
        browser.go_back()
    elif any(w in command for w in ("домой", "главная", "мультики", "на главную")):
        browser.go_home()
    elif any(w in command for w in ("найди", "включи", "поищи", "хочу")):
        if cfg["strict_whitelist"]:
            log.info("Свободный поиск отключен (strict_whitelist=true), а фраза не найдена в избранном.")
            return
        query = command
        for w in ("найди", "включи", "поищи", "хочу", "видео", "мультик про", "мультик"):
            query = query.replace(w, "")
        query = query.strip()
        if query:
            log.info("Ищу на YouTube Kids: %s", query)
            browser.search_kids(query)
            time.sleep(2)
            browser.click_first_result()
    else:
        log.info("Команда не распознана как известная.")


def main():
    cfg = load_config()
    log.info(
        "start_word='%s' | stop_word='%s' | auto_stop_after_s=%s | strict_whitelist=%s",
        cfg["start_word"], cfg["stop_word"], cfg["auto_stop_after_s"], cfg["strict_whitelist"],
    )

    browser = BrowserController(home_url=cfg["home_url"])

    model_size = os.environ.get("WHISPER_MODEL", "small")
    device = os.environ.get("WHISPER_DEVICE", "cpu")
    transcriber = Transcriber(model_size=model_size, device=device, language="ru")

    running = True
    active = False

    def shutdown(*_):
        nonlocal running
        log.info("Останавливаюсь...")
        running = False

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    log.info("Готово. Слушаю микрофон... (скажите '%s', чтобы начать)", cfg["start_word"])
    with Recorder() as recorder:
        while running:
            try:
                idle_timeout = cfg["auto_stop_after_s"] if active else None
                audio = recorder.listen_for_utterance(idle_timeout_s=idle_timeout)

                if audio is None:
                    # тишина дольше auto_stop_after_s в активном режиме
                    log.info("Тишина слишком долго - авто-остановка, жду '%s'", cfg["start_word"])
                    active = False
                    play_beep(freq=440)
                    continue

                text = transcriber.transcribe(audio)
                if not text:
                    continue
                log.debug("Распознано (сырое): %s", text)

                if not active:
                    if cfg["start_word"] not in text:
                        continue
                    active = True
                    play_beep(freq=1200)
                    log.info("Ассистент активирован")
                    remainder = text.replace(cfg["start_word"], "", 1).strip()
                    if remainder:
                        browser.restart_if_dead()
                        handle_command(browser, cfg, remainder)
                    continue

                if cfg["stop_word"] in text:
                    active = False
                    play_beep(freq=440)
                    log.info("Ассистент остановлен")
                    continue

                browser.restart_if_dead()
                handle_command(browser, cfg, text)
            except Exception as exc:
                log.exception("Ошибка в основном цикле: %s", exc)
                time.sleep(1)

    browser.quit()


if __name__ == "__main__":
    sys.exit(main())
