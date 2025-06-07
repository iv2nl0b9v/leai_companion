import time
import os
from dotenv import load_dotenv
from wake_word_detector_lib import WakeWordDetector


def main():
    # Load environment variables
    load_dotenv()

    # Get access key from environment variable
    ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")
    if not ACCESS_KEY:
        raise ValueError("Please set the PICOVOICE_ACCESS_KEY environment variable")

    # Print available keywords
    print("Available keywords:", WakeWordDetector.list_keywords())

    # Example custom callback function
    def on_wake_word(keyword):
        print(f"🎯 Wake word detected: {keyword}")

    try:
        # Initialize with some example keywords (you can choose from the available keywords)
        keywords = ["picovoice", "bumblebee"]

        # Create detector with custom callback and higher sensitivity
        with WakeWordDetector(
            access_key=ACCESS_KEY,
            keywords=keywords,
            sensitivities=[0.7] * len(keywords),
            callback=on_wake_word,
        ) as detector:
            print("\n🎤 Listening for wake words... (Press Ctrl+C to exit)")
            print(f"Try saying one of: {', '.join(keywords)}")

            # Keep the program running
            while True:
                time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n👋 Stopping...")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
