"""Запись фразы с микрофона по VAD (детектору голоса) и распознавание Faster-Whisper."""
import collections
import logging
import queue
import time

import numpy as np
import sounddevice as sd
import webrtcvad

log = logging.getLogger("audio")

SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000  # 480


def play_beep(freq=880, duration=0.12, volume=0.2):
    """Короткий звуковой сигнал обратной связи, что ассистент услышал wake word."""
    try:
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
        tone = np.sin(freq * 2 * np.pi * t) * volume
        sd.play(tone.astype(np.float32), SAMPLE_RATE)
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
        self._stream = sd.RawInputStream(
            samplerate=SAMPLE_RATE,
            blocksize=FRAME_SAMPLES,
            dtype="int16",
            channels=1,
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
        """Блокирующе ждёт речь, возвращает np.float32 массив [-1,1] 16kHz.

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
            is_speech = self.vad.is_speech(frame, SAMPLE_RATE)

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
        return audio


class Transcriber:
    def __init__(self, model_size="small", device="cpu", language="ru"):
        from faster_whisper import WhisperModel

        log.info("Загрузка модели Faster-Whisper '%s' (может занять время при первом запуске)...", model_size)
        compute_type = "int8" if device == "cpu" else "float16"
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self.language = language

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size < SAMPLE_RATE * 0.2:
            return ""
        segments, _ = self.model.transcribe(
            audio,
            language=self.language,
            vad_filter=False,
            beam_size=1,
        )
        return " ".join(seg.text.strip() for seg in segments).strip().lower()
