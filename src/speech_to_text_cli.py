"""
Speech-to-text command line interface using Vosk.

Example usage:
    # Using the small English model:
    python src/speech_to_text_cli.py --model vosk-model-small-en-us-0.15
"""

import argparse
from speech_to_text_lib import SpeechToText


def main():
    parser = argparse.ArgumentParser(description="Real-time Speech-to-Text using Vosk")
    parser.add_argument(
        "--model", type=str, default="model", help="Path to the Vosk model directory"
    )
    parser.add_argument("--device", type=int, help="Input device index")
    args = parser.parse_args()

    # List available audio devices
    print("Available audio input devices:")
    print(SpeechToText.list_audio_devices())
    print()

    # Define a callback for text processing
    def handle_text(text, is_partial):
        if is_partial:
            print(f"Partial: {text}", end="\r")
        else:
            print(f"\nRecognized: {text}")

    print("#" * 80)
    print("Press Ctrl+C to stop the recording")
    print("#" * 80)

    # Initialize and start speech recognition
    stt = SpeechToText(model=args.model, device=args.device)
    stt.process_audio(text_callback=handle_text)


if __name__ == "__main__":
    main()
