"""Обёртка над Selenium/Chromium для управления браузером."""
import logging
import os
import urllib.parse

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

log = logging.getLogger("browser")

CHROMIUM_BIN = "/usr/bin/chromium"
CHROMEDRIVER_BIN = "/usr/bin/chromedriver"


class BrowserController:
    def __init__(self, home_url: str, profile_dir: str = "/data/chrome-profile"):
        self.home_url = home_url
        self.driver = self._start_driver(profile_dir)

    def _start_driver(self, profile_dir: str):
        self._clear_stale_lock(profile_dir)
        options = webdriver.ChromeOptions()
        options.binary_location = CHROMIUM_BIN
        options.add_argument(f"--user-data-dir={profile_dir}")
        options.add_argument("--autoplay-policy=no-user-gesture-required")
        options.add_argument("--start-maximized")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--no-first-run")
        options.add_argument("--disable-infobars")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])

        service = Service(executable_path=CHROMEDRIVER_BIN)
        driver = webdriver.Chrome(service=service, options=options)
        driver.get(self.home_url)
        return driver

    @staticmethod
    def _clear_stale_lock(profile_dir: str):
        """Chrome оставляет Singleton*-файлы в profile_dir при нечистом завершении.

        Контейнер каждый раз стартует с чистого процесса - предыдущий Chromium
        точно мёртв, так что эти файлы всегда протухшие. Без очистки Chrome
        считает профиль "уже используется" и падает на каждом старте - контейнер
        уходит в бесконечный restart-loop (restart: unless-stopped перезапускает
        снова и снова, каждый раз с тем же результатом).
        """
        for name in ("SingletonLock", "SingletonSocket", "SingletonCookie"):
            try:
                os.remove(os.path.join(profile_dir, name))
            except FileNotFoundError:
                pass

    def restart_if_dead(self, profile_dir: str = "/data/chrome-profile"):
        try:
            _ = self.driver.title
        except Exception:
            log.warning("Chromium перестал отвечать, перезапускаю...")
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = self._start_driver(profile_dir)

    def go_home(self):
        self.driver.get(self.home_url)

    def open_url(self, url: str):
        self.driver.get(url)

    def search_kids(self, query: str):
        q = urllib.parse.quote_plus(query)
        self.driver.get(f"https://www.youtubekids.com/results?search_query={q}")

    def click_first_result(self):
        try:
            first = self.driver.find_element(By.CSS_SELECTOR, "a#video-title, a.compact-media-item-image")
            first.click()
        except Exception as exc:
            log.info("Не удалось кликнуть по первому результату: %s", exc)

    def toggle_play_pause(self):
        self._send_key(Keys.SPACE)

    def go_back(self):
        try:
            self.driver.back()
        except Exception:
            pass

    def fullscreen(self):
        self._send_key("f")

    def volume_up(self):
        self._send_key(Keys.ARROW_UP)

    def volume_down(self):
        self._send_key(Keys.ARROW_DOWN)

    def _send_key(self, key):
        try:
            self.driver.find_element(By.TAG_NAME, "body").send_keys(key)
        except Exception as exc:
            log.debug("send_keys failed: %s", exc)

    def quit(self):
        try:
            self.driver.quit()
        except Exception:
            pass
