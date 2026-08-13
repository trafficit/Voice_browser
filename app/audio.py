"""Запись фразы с микрофона по VAD (детектору голоса) и распознавание Faster-Whisper."""
import collections
import io
import logging
import os
import queue
import subprocess
import time
import wave

import numpy as np
import sounddevice as sd
import webrtcvad

log = logging.getLogger("audio")

# Пишем на "родной" частоте устройства (её умеет почти любое USB-аудио) -
# внутри контейнера ALSA работает напрямую с железом в обход PulseAudio,
# который на хосте незаметно делает ресемплинг сам. 16000 Гц многие
# устройства не поддерживают напрямую, поэтому пишем на 48000 и сами
# понижаем частоту перед распознаванием (Whisper ждёт именно 16000 Гц).
REC_SAMPLE_RATE = int(os.environ.get("AUDIO_SAMPLE_RATE", "48000"))
WHISPER_SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = REC_SAMPLE_RATE * FRAME_MS // 1000

# Подстрока для выбора нужного микрофона среди нескольких устройств
# (по умолчанию ищем USB-микрофон, а не встроенный в материнку).
AUDIO_DEVICE_MATCH = os.environ.get("AUDIO_DEVICE_MATCH", "usb")

# Частота воспроизведения гудков/озвучки - как и с записью, лучше зафиксировать
# на безопасном значении и ресемплировать самим, чем упереться в
# "Invalid sample rate" на конкретной звуковой карте.
OUTPUT_SAMPLE_RATE = int(os.environ.get("AUDIO_OUTPUT_SAMPLE_RATE", "48000"))
# Пусто = устройство вывода по умолчанию. Если звук идёт не туда (например
# в HDMI вместо колонок) - впишите часть названия нужного устройства,
# например "PCH" (см. список устройств в логах при старте).
AUDIO_OUTPUT_MATCH = os.environ.get("AUDIO_OUTPUT_MATCH", "")


_device_cache = {}


def _resolve_device(name_substring: str, kind: str):
    """kind: 'input' или 'output'. Результат кэшируется - список устройств
    запрашивается только один раз за это сочетание маски/типа."""
    cache_key = (name_substring, kind)
    if cache_key in _device_cache:
        return _device_cache[cache_key]

    channels_key = f"max_{kind}_channels"
    idx = None
    if name_substring:
        try:
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if dev.get(channels_key, 0) > 0 and name_substring.lower() in dev["name"].lower():
                    idx = i
                    break
        except Exception as exc:
            log.warning("Не удалось получить список аудиоустройств: %s", exc)

    _device_cache[cache_key] = idx
    return idx


def log_audio_devices():
    """Печатает список всех аудиоустройств и что выбрано - для диагностики."""
    try:
        devices = sd.query_devices()
    except Exception as exc:
        log.warning("Не удалось получить список аудиоустройств: %s", exc)
        return

    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0 or dev.get("max_output_channels", 0) > 0:
            log.info(
                "Аудиоустройство [%d]: %s (вход=%d, выход=%d)",
                idx, dev["name"], dev.get("max_input_channels", 0), dev.get("max_output_channels", 0),
            )

    in_idx = _resolve_device(AUDIO_DEVICE_MATCH, "input")
    out_idx = _resolve_device(AUDIO_OUTPUT_MATCH, "output")
    log.info("Микрофон: %s", devices[in_idx]["name"] if in_idx is not None else "устройство по умолчанию")
    log.info("Вывод звука: %s", devices[out_idx]["name"] if out_idx is not None else "устройство по умолчанию")


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr or audio.size == 0:
        return audio
    target_len = int(round(audio.size * target_sr / orig_sr))
    orig_idx = np.arange(audio.size)
    target_idx = np.linspace(0, audio.size - 1, num=target_len)
    return np.interp(target_idx, orig_idx, audio).astype(np.float32)


def _play(samples: np.ndarray, sr: int):
    samples = _resample(samples, sr, OUTPUT_SAMPLE_RATE)
    device = _resolve_device(AUDIO_OUTPUT_MATCH, "output")
    sd.play(samples, OUTPUT_SAMPLE_RATE, device=device)
    sd.wait()


def speak(text: str):
    """Голосовая обратная связь через офлайн-синтез espeak-ng.

    espeak-ng генерирует WAV в память (--stdout), а воспроизводим его тем же
    путём (sounddevice), что и гудки play_beep - так надёжнее, чем полагаться
    на собственный аудио-бэкенд espeak-ng внутри контейнера.
    """
    try:
        proc = subprocess.run(
            ["espeak-ng", "-v", "ru", "-s", "165", "--stdout", text],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=10,
        )
        with wave.open(io.BytesIO(proc.stdout), "rb") as wf:
            sr = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        _play(samples, sr)
    except Exception:
        log.warning("Не удалось озвучить '%s'", text, exc_info=True)


def play_beep(freq=880, duration=0.12, volume=0.2):
    """Короткий звуковой сигнал обратной связи, что ассистент услышал wake word."""
    try:
        t = np.linspace(0, duration, int(OUTPUT_SAMPLE_RATE * duration), False)
        tone = (np.sin(freq * 2 * np.pi * t) * volume).astype(np.float32)
        _play(tone, OUTPUT_SAMPLE_RATE)
    except Exception:
        log.warning("Не удалось воспроизвести гудок", exc_info=True)


class Recorder:
    """Слушает микрофон непрерывно и отдаёт фразы, как только обнаружена пауза после речи."""

    def __init__(self, vad_aggressiveness=2, max_silence_ms=800, max_duration_s=8, min_speech_ms=250):
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.max_silence_frames = max_silence_ms // FRAME_MS
        self.max_frames = max_duration_s * 1000 // FRAME_MS
        self.min_speech_frames = min_speech_ms // FRAME_MS
        self._q = queue.Queue()

        log_audio_devices()
        device = _resolve_device(AUDIO_DEVICE_MATCH, "input")
        self._stream = sd.RawInputStream(
            samplerate=REC_SAMPLE_RATE,
            blocksize=FRAME_SAMPLES,
            dtype="int16",
            channels=1,
            device=device,
            callback=self._callback,
        )

    def _callback(self, indata, frames, time_info, status):
        if status:
            log.debug("audio status: %s", status)
        self._q.put(bytes(indata))

    def __enter__(self):
        self._stream.start()
        return self

    def __exit__(self, *exc):
        self._stream.stop()
        self._stream.close()

    def listen_for_utterance(self, idle_timeout_s=None):
        """Блокирующе ждёт речь, возвращает np.float32 массив [-1,1] на 16kHz.

        Если задан idle_timeout_s и за это время не появилось ни одной фразы -
        возвращает None (используется для авто-остановки ассистента по тишине).
        """
        ring = collections.deque(maxlen=self.max_silence_frames)
        voiced_frames = []
        triggered = False
        wait_start = time.monotonic()

        while True:
            frame = self._q.get()
            if len(frame) != FRAME_SAMPLES * 2:
                continue
            is_speech = self.vad.is_speech(frame, REC_SAMPLE_RATE)

            if not triggered:
                ring.append((frame, is_speech))
                num_voiced = len([f for f, s in ring if s])
                if num_voiced > 0.6 * ring.maxlen and len(ring) >= max(self.min_speech_frames, 3):
                    triggered = True
                    voiced_frames.extend(f for f, s in ring)
                    ring.clear()
                elif idle_timeout_s is not None and (time.monotonic() - wait_start) > idle_timeout_s:
                    return None
            else:
                voiced_frames.append(frame)
                ring.append((frame, is_speech))
                num_unvoiced = len([f for f, s in ring if not s])
                if num_unvoiced > 0.9 * ring.maxlen or len(voiced_frames) > self.max_frames:
                    break

        pcm = b"".join(voiced_frames)
        audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        return _resample(audio, REC_SAMPLE_RATE, WHISPER_SAMPLE_RATE)


class Transcriber:
    def __init__(self, model_size="small", device="cpu", language="ru"):
        from faster_whisper import WhisperModel

        log.info("Загрузка модели Faster-Whisper '%s' (может занять время при первом запуске)...", model_size)
        compute_type = "int8" if device == "cpu" else "float16"
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.language = language

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size < WHISPER_SAMPLE_RATE * 0.2:
            return ""
        segments, _ = self.model.transcribe(
            audio,
            language=self.language,
            vad_filter=False,
            beam_size=1,
        )
        return " ".join(seg.text.strip() for seg in segments).strip().lower()
