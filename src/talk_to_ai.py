"""
AI Companion with Voice Interaction

This script runs an AI companion that listens for a wake word and then
transcribes your voice command to interact with a Gemini model.

Usage:
    - To run the AI companion:
        python src/talk_to_ai.py --wake_keyword "your-wake-word" [optional arguments]

    - To list available audio devices:
        python src/talk_to_ai.py --list-devices

Optional Arguments:
    --device DEVICE_INDEX         : Index of the audio input device to use.
    --latency SECONDS             : Audio stream latency (default: 0.1).
    --gemini_model MODEL_NAME     : Gemini model to use (default: gemini-2.5-flash-preview-05-20).

Example:
    python src/talk_to_ai.py --wake_keyword "bumblebee" --device 1
"""

import os
import argparse
import threading
import re
from typing import Optional, NoReturn, Union, Dict, Any, Literal
import google.generativeai as genai
from dotenv import load_dotenv
from wake_word_detector_lib import WakeWordDetector
from google_cloud_speech_lib import SpeechRecognizer
from google_cloud_tts_lib import TextToSpeech
import time


class AICompanion:
    def __init__(
        self,
        wake_keyword: str,
        device: Optional[Union[int, str, Dict[str, Any]]] = None,
        latency: float = 0.1,
        model_name: str = "gemini-2.5-flash-preview-05-20",
    ) -> None:
        """Initialize the AI companion.

        Args:
            wake_keyword (str): Wake keyword to listen for
            device (Union[int, str, dict], optional): Audio input device (index, name, or dict)
            latency (float): Audio stream latency in seconds
            model_name (str): Name of the Gemini model to use
        """
        load_dotenv()

        # Initialize Gemini
        api_key: Optional[str] = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        genai.configure(api_key=api_key)
        print(f"\nInitializing Gemini with model: {model_name}")
        self.model: genai.GenerativeModel = genai.GenerativeModel(model_name)
        self.chat: genai.ChatSession = self.model.start_chat(history=[])

        # Initialize wake word detector
        access_key: Optional[str] = os.getenv("PICOVOICE_ACCESS_KEY")
        if not api_key:
            raise ValueError("PICOVOICE_ACCESS_KEY not found in environment variables")
        self.wake_detector = WakeWordDetector(
            access_key=access_key,
            keywords=[wake_keyword],
            sensitivities=[0.7],
            callback=self.on_wake_word,
            device=device,
            latency=latency,
        )

        # Initialize speech recognition
        device_index = device if isinstance(device, int) else None
        self.speech_recognizer = SpeechRecognizer(device_index=device_index)
        self.listening_for_command = False
        self.command_thread: Optional[threading.Thread] = None

        # Initialize Text-to-Speech
        self.tts = TextToSpeech()

    def on_wake_word(self, keyword: str) -> None:
        """Called when wake word is detected."""
        if not self.listening_for_command:
            print("\nWake word detected! Listening for your command...")
            self.listening_for_command = True
            self.command_thread = threading.Thread(target=self.listen_for_command)
            self.command_thread.start()

    def listen_for_command(self) -> None:
        """Listen for and process voice commands in a continuous conversation."""
        try:
            while self.listening_for_command:
                command_text: Optional[str] = None
                print("\nListening...")

                with self.speech_recognizer:
                    for (
                        transcript,
                        is_final,
                    ) in self.speech_recognizer.recognize_stream():
                        if is_final and transcript.strip():
                            command_text = transcript.strip()
                            print(f"\r{' ' * 80}\r", end="")
                            print(f"You: {command_text}")
                            break
                        elif not is_final and transcript.strip():
                            print(
                                f"\rYou (thinking...): {transcript}", end="", flush=True
                            )

                if command_text:
                    if command_text.lower().strip() in ["goodbye", "exit", "stop"]:
                        print("\nAI: Goodbye!")
                        self.tts.speak("Goodbye!")
                        self.tts.wait()
                        break

                    response: genai.GenerateContentResponse = self.chat.send_message(
                        command_text, stream=True
                    )

                    print("\nAI: ", end="")
                    response_text: str = ""
                    sentence_buffer: str = ""
                    for chunk in response:
                        chunk_text: str = chunk.text
                        print(chunk_text, end="", flush=True)
                        response_text += chunk_text
                        sentence_buffer += chunk_text

                        if any(p in sentence_buffer for p in ".!?"):
                            sentences = re.split(r"(?<=[.!?])\s*", sentence_buffer)
                            for sentence in sentences[:-1]:
                                if sentence.strip():
                                    self.tts.speak(sentence.strip())
                            sentence_buffer = sentences[-1]

                    if sentence_buffer.strip():
                        self.tts.speak(sentence_buffer.strip())

                    self.tts.wait()
                    print()
                else:
                    print("Did not catch that. Please try again.")

        except Exception as e:
            print(f"An error occurred during the conversation: {e}")
        finally:
            self.listening_for_command = False
            print("\nConversation ended. Say the wake word to start again.")

    def run(self) -> NoReturn:
        """Run the AI companion."""
        print(f"AI Companion is ready! Say the wake word to begin...")
        try:
            with self.wake_detector, self.tts:
                while True:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nShutting down AI Companion...")


def list_audio_devices() -> None:
    """List all available audio input devices."""
    print("\nListing all audio input devices:")
    WakeWordDetector.list_audio_devices()


def main() -> None:
    """Main entry point for the AI companion application."""
    parser = argparse.ArgumentParser(description="AI Companion with voice interaction")

    # Create argument groups
    required_args = parser.add_argument_group("required arguments")
    optional_args = parser.add_argument_group("optional arguments")

    # Required arguments (only if not listing devices)
    required_args.add_argument("--wake_keyword", help="Wake keyword to listen for")

    # Optional arguments
    optional_args.add_argument("--device", type=int, help="Audio input device index")
    optional_args.add_argument(
        "--latency", type=float, default=0.1, help="Audio stream latency in seconds"
    )
    optional_args.add_argument(
        "--list-devices",
        action="store_true",
        help="List available audio devices and exit",
    )
    optional_args.add_argument(
        "--gemini_model",
        type=str,
        default="gemini-2.5-flash-preview-05-20",
        help="Gemini model to use (default: %(default)s)",
    )

    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    # Validate required arguments if not just listing devices
    if not args.wake_keyword:
        if not args.list_devices:
            parser.error(
                "--wake_keyword is required unless --list-devices is specified"
            )
        return

    companion = AICompanion(
        wake_keyword=args.wake_keyword,
        device=args.device,
        latency=args.latency,
        model_name=args.gemini_model,
    )
    companion.run()


if __name__ == "__main__":
    main()
