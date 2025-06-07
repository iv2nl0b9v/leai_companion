import queue
import sys
import json
import sounddevice as sd
from sounddevice import CallbackFlags
from vosk import Model, KaldiRecognizer
from typing import Any, Optional, Union, Dict, Callable
import numpy as np
import time as time_lib


class SpeechToText:
    def __init__(
        self,
        model: str = "model",
        device: Optional[Union[int, str, Dict[str, Any]]] = None,
        sample_rate: int = 16000,
        latency: float = 0.1,
    ):
        """Initialize the speech-to-text engine.

        Args:
            model (str): Name of the Vosk model
            device (Union[int, str, dict], optional): Audio input device (index, name, or dict)
            sample_rate (int): Audio sample rate in Hz
            latency (float): Audio stream latency in seconds
        """
        try:
            self.model = Model(model_name=model)
            self.sample_rate = sample_rate
            self.device = self._get_device_id(device)
            self.latency = latency
            self.recognizer = KaldiRecognizer(self.model, self.sample_rate)
            self.q = queue.Queue()
            self.error_count = 0
            self.last_error_time = 0
            self.MAX_ERRORS = 5
            self.ERROR_RESET_TIME = 60  # seconds

            # Print audio device info for debugging
            print("\nSpeech Recognition Audio Configuration:")
            print(f"Using device: {sd.query_devices(self.device)}")
            print(f"Sample rate: {self.sample_rate}")
            print(f"Latency: {self.latency}s")

        except Exception as e:
            print(f"Error initializing speech recognition: {str(e)}")
            sys.exit(1)

    def _get_device_id(
        self, device: Optional[Union[int, str, Dict[str, Any]]] = None
    ) -> int:
        """Get the device ID for the audio input device."""
        try:
            if device is None:
                # Try to find a working input device
                devices = sd.query_devices()
                for i, dev in enumerate(devices):
                    if dev["max_input_channels"] > 0:
                        print(f"Selected input device {i}: {dev['name']}")
                        return i
                raise RuntimeError("No working input device found")
            elif isinstance(device, (int, str)):
                return sd.query_devices(device).get("index", 0)
            elif isinstance(device, dict):
                return device.get("index", 0)
            else:
                raise ValueError(f"Invalid device specification: {device}")
        except Exception as e:
            print(f"Error selecting audio device: {e}")
            print("Available devices:")
            print(sd.query_devices())
            raise

    def callback(
        self, indata: np.ndarray, frames: int, time: Any, status: CallbackFlags
    ) -> None:
        """Callback for audio stream processing"""
        current_time = time_lib.time()

        if status:
            if status.input_overflow:
                self.error_count += 1
                if current_time - self.last_error_time > self.ERROR_RESET_TIME:
                    self.error_count = 1
                self.last_error_time = current_time

                if self.error_count >= self.MAX_ERRORS:
                    print(
                        "\nToo many input overflows. Try increasing latency or using a different audio device."
                    )
                    return
            print(f"Audio callback status: {status}")
            return

        try:
            self.q.put(bytes(indata))
        except Exception as e:
            print(f"Error in audio callback: {e}")

    def process_audio(
        self, text_callback: Optional[Callable[[str, bool], None]] = None
    ) -> None:
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
                latency=self.latency,
            ):
                print("\nListening for speech...")
                while True:
                    try:
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
                    except queue.Empty:
                        continue
                    except json.JSONDecodeError as e:
                        print(f"Error decoding recognition result: {e}")
                    except Exception as e:
                        print(f"Error processing recognition data: {e}")

        except KeyboardInterrupt:
            print("\nStopping speech recognition.")
        except Exception as e:
            print(f"Error during speech recognition: {str(e)}")

    @staticmethod
    def list_audio_devices() -> None:
        """List all available audio input devices with detailed information."""
        devices = sd.query_devices()
        input_devices = []
        print("\nAvailable Audio Input Devices:")
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0:
                print(f"\nDevice {i}: {dev['name']}")
                print(f"  Channels: {dev['max_input_channels']}")
                print(f"  Sample rates: {dev['default_samplerate']}")
                input_devices.append(i)
        return input_devices
