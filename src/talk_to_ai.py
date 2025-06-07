import os
import argparse
import threading
from typing import Optional, NoReturn, Union, Dict, Any, Literal
import google.generativeai as genai
from dotenv import load_dotenv
from wake_word_detector_lib import WakeWordDetector
from speech_to_text_lib import SpeechToText
import time


class AICompanion:
    def __init__(
        self,
        vosk_model: str,
        wake_keyword: str,
        device: Optional[Union[int, str, Dict[str, Any]]] = None,
        latency: float = 0.1,
        model_name: str = "gemini-2.5-flash-preview-05-20",
    ) -> None:
        """Initialize the AI companion.

        Args:
            vosk_model (str): Path to the Vosk model
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
        self.speech_to_text = SpeechToText(
            model=vosk_model, device=device, latency=latency
        )
        self.listening_for_command = False
        self.command_thread: Optional[threading.Thread] = None

    def on_wake_word(self, keyword: str) -> None:
        """Called when wake word is detected."""
        if not self.listening_for_command:
            print("\nWake word detected! Listening for your command...")
            self.listening_for_command = True
            self.command_thread = threading.Thread(target=self.listen_for_command)
            self.command_thread.start()

    def on_speech_text(self, text: str, is_partial: bool) -> None:
        """Handle recognized speech text."""
        if not is_partial and text.strip():
            print(f"\nYou: {text}")

            # Get AI response
            response: genai.GenerateContentResponse = self.chat.send_message(
                text, stream=True
            )

            # Print AI response
            print("\nAI: ", end="")
            response_text: str = ""
            for chunk in response:
                chunk_text: str = chunk.text
                print(chunk_text, end="", flush=True)
                response_text += chunk_text
            print("\n")

            # Stop listening after processing the command
            self.listening_for_command = False

    def listen_for_command(self) -> None:
        """Listen for and process voice commands."""
        self.speech_to_text.process_audio(self.on_speech_text)

    def run(self) -> NoReturn:
        """Run the AI companion."""
        print(f"AI Companion is ready! Say the wake word to begin...")
        try:
            with self.wake_detector:
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
    required_args.add_argument("--vosk_model", help="Path to Vosk model")
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
        choices=["gemini-pro", "gemini-pro-vision", "gemini-2.5-flash-preview-05-20"],
        default="gemini-2.5-flash-preview-05-20",
        help="Gemini model to use (default: %(default)s)",
    )

    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        return

    # Validate required arguments if not just listing devices
    if not args.vosk_model or not args.wake_keyword:
        if not args.list_devices:
            parser.error(
                "--vosk_model and --wake_keyword are required unless --list-devices is specified"
            )
        return

    companion = AICompanion(
        vosk_model=args.vosk_model,
        wake_keyword=args.wake_keyword,
        device=args.device,
        latency=args.latency,
        model_name=args.gemini_model,
    )
    companion.run()


if __name__ == "__main__":
    main()
