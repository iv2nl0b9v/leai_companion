import pvporcupine
import sounddevice as sd
import numpy as np
from typing import Callable, Optional, List, Dict, Any, Union
import time as time_lib


class WakeWordDetector:
    def __init__(
        self,
        access_key: str,
        keywords: List[str],
        sensitivities: Optional[List[float]] = None,
        callback: Optional[Callable[[str], None]] = None,
        device: Optional[Union[int, str, Dict[str, Any]]] = None,
        latency: float = 0.1,
    ):
        """
        Initialize the wake word detector.

        Args:
            access_key (str): Picovoice access key
            keywords (List[str]): List of keywords to detect
            sensitivities (List[float], optional): Detection sensitivity for each keyword (0-1)
            callback (Callable[[str], None], optional): Function to call when wake word is detected
            device (Union[int, str, dict], optional): Audio input device (index, name, or dict)
            latency (float): Audio stream latency in seconds
        """
        if sensitivities is None:
            sensitivities = [0.5] * len(keywords)

        self.keywords = keywords
        self.callback = callback or (lambda x: print(f"Wake word detected: {x}"))
        self.device = self._get_device_id(device)
        self.error_count = 0
        self.last_error_time = 0
        self.MAX_ERRORS = 5
        self.ERROR_RESET_TIME = 60  # seconds

        try:
            self.porcupine = pvporcupine.create(
                access_key=access_key, keywords=keywords, sensitivities=sensitivities
            )

            # Print audio device info for debugging
            print("\nAudio Device Configuration:")
            print(f"Using device: {sd.query_devices(self.device)}")
            print(f"Sample rate: {self.porcupine.sample_rate}")
            print(f"Frame length: {self.porcupine.frame_length}")
            print(f"Latency: {latency}s")

            self.audio_stream = sd.InputStream(
                samplerate=self.porcupine.sample_rate,
                device=self.device,
                channels=1,
                dtype=np.int16,
                blocksize=self.porcupine.frame_length,
                callback=self._audio_callback,
                latency=latency,
            )

        except Exception as e:
            raise RuntimeError(f"Failed to initialize wake word detector: {str(e)}")

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

    def _audio_callback(self, indata, frames, time, status):
        """Handle audio input data."""
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
            pcm = indata.flatten().astype(np.int16)
            keyword_index = self.porcupine.process(pcm)

            if keyword_index >= 0:
                detected_keyword = self.keywords[keyword_index]
                self.callback(detected_keyword)
        except Exception as e:
            print(f"Error processing audio data: {e}")

    def start(self):
        """Start listening for wake words."""
        self.audio_stream.start()

    def stop(self):
        """Stop listening for wake words."""
        try:
            if hasattr(self, "audio_stream"):
                self.audio_stream.stop()
        except Exception as e:
            print(f"Error stopping audio stream: {e}")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        if hasattr(self, "porcupine"):
            self.porcupine.delete()

    @staticmethod
    def list_keywords():
        """List all available built-in keywords."""
        return pvporcupine.KEYWORDS

    @staticmethod
    def list_audio_devices():
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
