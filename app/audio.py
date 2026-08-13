"""Запись фразы с микрофона по VAD (детектору голоса) и распознавание Faster-Whisper."""
import collections
import logging
import os
import queue
import time

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


def _find_input_device(name_substring: str):
    if not name_substring:
        return None
    try:
        devices = sd.query_devices()
    except Exception as exc:
        log.warning("Не удалось получить список аудиоустройств: %s", exc)
        return None

    for idx, dev in enumerate(devices):
        if dev.get("max_input_channels", 0) > 0 and name_substring.lower() in dev["name"].lower():
            log.info("Микрофон выбран: [%d] %s", idx, dev["name"])
            return idx

    log.warning("Микрофон по маске '%s' не найден, использую устройство по умолчанию", name_substring)
    for idx, dev in enumerate(devices):
        log.info("Доступное аудиоустройство [%d]: %s (входов=%d)", idx, dev["name"], dev.get("max_input_channels", 0))
    return None


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr or audio.size == 0:
        return audio
    target_len = int(round(audio.size * target_sr / orig_sr))
    orig_idx = np.arange(audio.size)
    target_idx = np.linspace(0, audio.size - 1, num=target_len)
    return np.interp(target_idx, orig_idx, audio).astype(np.float32)


def play_beep(freq=880, duration=0.12, volume=0.2):
    """Короткий звуковой сигнал обратной связи, что ассистент услышал wake word."""
    try:
        t = np.linspace(0, duration, int(REC_SAMPLE_RATE * duration), False)
        tone = np.sin(freq * 2 * np.pi * t) * volume
        sd.play(tone.astype(np.float32), REC_SAMPLE_RATE)
    except Exception as exc:
        log.debug("beep failed: %s", exc)


class Recorder:
    """Слушает микрофон непрерывно и отдаёт фразы, как только обнаружена пауза после речи."""

    def __init__(self, vad_aggressiveness=2, max_silence_ms=800, max_duration_s=8, min_speech_ms=250):
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.max_silence_frames = max_silence_ms // FRAME_MS
        self.max_frames = max_duration_s * 1000 // FRAME_MS
        self.min_speech_frames = min_speech_ms // FRAME_MS
        self._q = queue.Queue()

        device = _find_input_device(AUDIO_DEVICE_MATCH)
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
