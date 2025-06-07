import pvporcupine
import sounddevice as sd
import numpy as np
from typing import Callable, Optional, List


class WakeWordDetector:
    def __init__(
        self,
        access_key: str,
        keywords: List[str],
        sensitivities: Optional[List[float]] = None,
        callback: Optional[Callable[[str], None]] = None,
    ):
        """
        Initialize the wake word detector.

        Args:
            access_key (str): Picovoice access key
            keywords (List[str]): List of keywords to detect
            sensitivities (List[float], optional): Detection sensitivity for each keyword (0-1)
            callback (Callable[[str], None], optional): Function to call when wake word is detected
        """
        if sensitivities is None:
            sensitivities = [0.5] * len(keywords)

        self.keywords = keywords
        self.callback = callback or (lambda x: print(f"Wake word detected: {x}"))

        try:
            self.porcupine = pvporcupine.create(
                access_key=access_key, keywords=keywords, sensitivities=sensitivities
            )

            self.audio_stream = sd.InputStream(
                samplerate=self.porcupine.sample_rate,
                channels=1,
                dtype=np.int16,
                blocksize=self.porcupine.frame_length,
                callback=self._audio_callback,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to initialize wake word detector: {str(e)}")

    def _audio_callback(self, indata, frames, time, status):
        """Handle audio input data."""
        if status:
            print(f"Audio callback status: {status}")
            return

        pcm = indata.flatten().astype(np.int16)
        keyword_index = self.porcupine.process(pcm)

        if keyword_index >= 0:
            detected_keyword = self.keywords[keyword_index]
            self.callback(detected_keyword)

    def start(self):
        """Start listening for wake words."""
        self.audio_stream.start()

    def stop(self):
        """Stop listening for wake words."""
        self.audio_stream.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        self.porcupine.delete()

    @staticmethod
    def list_keywords():
        """List all available built-in keywords."""
        return pvporcupine.KEYWORDS
