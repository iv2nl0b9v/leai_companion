import pyaudio
from google.cloud import speech
from dotenv import load_dotenv


class SpeechRecognizer:
    """A class to handle streaming speech recognition."""

    def __init__(
        self,
        rate: int = 16000,
        chunk: int = 1600,
        language_code: str = "en-US",
        device_index: int | None = None,
    ):
        """Initializes the speech recognizer."""
        self.rate = rate
        self.chunk = chunk
        self.language_code = language_code
        self.client = None
        self.streaming_config = None
        self.audio = None
        self.stream = None
        self.device_index = device_index

    def __enter__(self):
        """Sets up the speech client and audio stream."""
        load_dotenv()
        self.client = speech.SpeechClient()

        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=self.rate,
            language_code=self.language_code,
        )

        self.streaming_config = speech.StreamingRecognitionConfig(
            config=config, interim_results=True
        )

        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
            input_device_index=self.device_index,
        )
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Tears down the audio stream and PyAudio instance."""
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.audio:
            self.audio.terminate()

    def _audio_generator(self):
        """A generator that yields audio chunks from the microphone."""
        while self.stream and not self.stream.is_stopped():
            data = self.stream.read(self.chunk, exception_on_overflow=False)
            yield data

    def recognize_stream(self):
        """
        Recognizes speech from the microphone stream and yields transcripts.
        Yields:
            tuple: A tuple containing the transcript (str) and a boolean
                   indicating if the result is final (bool).
        """
        requests = (
            speech.StreamingRecognizeRequest(audio_content=content)
            for content in self._audio_generator()
        )

        responses = self.client.streaming_recognize(self.streaming_config, requests)

        for response in responses:
            if not response.results:
                continue

            result = response.results[0]
            if not result.alternatives:
                continue

            transcript = result.alternatives[0].transcript

            yield transcript, result.is_final
