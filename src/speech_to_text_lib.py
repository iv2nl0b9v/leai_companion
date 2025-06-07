import queue
import sys
import json
import sounddevice as sd
from sounddevice import CallbackFlags
from vosk import Model, KaldiRecognizer
from typing import Any
import numpy as np


class SpeechToText:
    def __init__(self, model="model", device=None, sample_rate=16000):
        """Initialize the speech-to-text engine.

        Args:
            model (str): Name of the Vosk model
            device (int, optional): Input device index. None for default device.
            sample_rate (int): Audio sample rate in Hz
        """
        try:
            self.model = Model(model_name=model)
            self.sample_rate = sample_rate
            self.device = device
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
            self.q = queue.Queue()

        except Exception as e:
            print(f"Error initializing speech recognition: {str(e)}")
            sys.exit(1)

    def callback(
        self, indata: np.ndarray, frames: int, time: Any, status: CallbackFlags
    ):
        """Callback for audio stream processing"""
        if status:
            print(status)
        self.q.put(bytes(indata))

    def process_audio(self, text_callback=None):
        """Process audio stream and perform real-time transcription

        Args:
            text_callback (callable, optional): Callback function that receives two parameters:
                - text (str): The recognized text
                - is_partial (bool): Whether this is a partial or final result
        """
        try:
            with sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=8000,
                device=self.device,
                dtype="int16",
                channels=1,
                callback=self.callback,
            ):

                while True:
                    data = self.q.get()
                    if self.recognizer.AcceptWaveform(data):
                        result = json.loads(self.recognizer.Result())
                        if result["text"]:
                            if text_callback:
                                text_callback(result["text"], False)
                            else:
                                print(f"Recognized: {result['text']}")
                    else:
                        partial = json.loads(self.recognizer.PartialResult())
                        if partial["partial"]:
                            if text_callback:
                                text_callback(partial["partial"], True)
                            else:
                                print(f"Partial: {partial['partial']}", end="\r")

        except KeyboardInterrupt:
            print("\nStopping speech recognition.")
        except Exception as e:
            print(f"Error during speech recognition: {str(e)}")

    @staticmethod
    def list_audio_devices():
        """List all available audio input devices

        Returns:
            str: Formatted string containing device information
        """
        return str(sd.query_devices())
