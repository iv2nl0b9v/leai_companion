import pyaudio
from google.cloud import texttospeech
from dotenv import load_dotenv
import threading
import queue


class TextToSpeech:
    """A class to handle streaming Text-to-Speech conversion and audio playback."""

    def __init__(
        self,
        language_code="en-US",
        voice_name="en-US-Wavenet-D",
        speaking_rate=1.0,
    ):
        """Initializes the Text-to-Speech client."""
        load_dotenv()
        self.client = texttospeech.TextToSpeechClient()
        self.voice = texttospeech.VoiceSelectionParams(
            language_code=language_code, name=voice_name
        )
        self.audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            speaking_rate=speaking_rate,
            sample_rate_hertz=24000,
        )
        self.audio = None
        self.stream = None
        self.audio_queue = queue.Queue()
        self.player_thread = None
        self.playing = threading.Event()

    def __enter__(self):
        """Sets up the audio stream and player thread."""
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=self.audio.get_format_from_width(2),
            channels=1,
            rate=24000,
            output=True,
        )
        self.playing.set()
        self.player_thread = threading.Thread(target=self._play_audio)
        self.player_thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Tears down the player thread and audio stream."""
        self.playing.clear()
        self.audio_queue.put(None)  # Sentinel to unblock queue.get()
        if self.player_thread:
            self.player_thread.join()
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()

    def _play_audio(self):
        """Pulls audio from queue and plays it."""
        while self.playing.is_set():
            try:
                audio_chunk = self.audio_queue.get(timeout=0.1)
                if audio_chunk is None:
                    self.audio_queue.task_done()
                    break
                self.stream.write(audio_chunk)
                self.audio_queue.task_done()
            except queue.Empty:
                continue

    def speak(self, text: str):
        """Synthesizes speech from text and adds it to the playback queue."""
        if not text:
            return

        synthesis_input = texttospeech.SynthesisInput(text=text)
        response = self.client.synthesize_speech(
            input=synthesis_input, voice=self.voice, audio_config=self.audio_config
        )
        self.audio_queue.put(response.audio_content)

    def wait(self):
        """Blocks until the audio queue is empty."""
        self.audio_queue.join()
