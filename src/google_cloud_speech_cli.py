"""
Google Cloud Speech-to-Text command-line interface.

Example usage:
    python src/google_cloud_speech_cli.py
"""

from google_cloud_speech_lib import SpeechRecognizer


def main():
    """Streams audio from the microphone and prints real-time transcriptions."""
    try:
        with SpeechRecognizer() as recognizer:
            print("Listening... Press Ctrl+C to stop.")
            for transcript, is_final in recognizer.recognize_stream():
                if is_final:
                    print(f"Final: {transcript}")
                else:
                    # Use a carriage return to overwrite the previous interim result.
                    print(f"Interim: {transcript}\r", end="")

    except KeyboardInterrupt:
        print("\nStopping...")


if __name__ == "__main__":
    main()
