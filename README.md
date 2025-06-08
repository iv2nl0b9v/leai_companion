# LeAI Companion

An AI companion for Leia that uses wake word detection to enable voice-activated interactions.

## Overview

LeAI Companion uses Picovoice's Porcupine wake word detection library to listen for specific keywords, allowing for hands-free activation of your AI companion. When activated, it uses speech recognition to understand your commands and communicates with Google's Gemini AI to provide intelligent responses.

## Prerequisites

- Python 3.7 or later
- A microphone connected to your computer
- Picovoice Access Key (get it from [Picovoice Console](https://console.picovoice.ai/))
- Google API Key (get it from [Google AI Studio](https://makersuite.google.com/app/apikey))

## Setup and Installation

1. Clone this repository:
   ```bash
   git clone [repository-url]
   cd leai_companion
   ```

2. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux/MacOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root and add your API keys:
   ```
   PICOVOICE_ACCESS_KEY=your_picovoice_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   ```

## Usage of the wake word example

1. Ensure your virtual environment is activated:
   ```bash
   # Windows
   .\venv\Scripts\activate

   # Linux/MacOS
   source venv/bin/activate
   ```

2. Run the main script:
   ```bash
   python src/wake_word_detector_cli.py
   ```

3. The program will start listening for wake words. By default, it listens for configured wake words that activate your AI companion.

4. To stop the program, press Ctrl+C.

## Speech-to-Text

The companion uses speech recognition to convert your voice into text. This enables natural voice interactions with the AI assistant. The system automatically starts listening for your voice input after detecting the wake word.

To ensure optimal speech recognition:
- Speak clearly and at a normal pace
- Use a good quality microphone
- Minimize background noise

To run the speech-to-text example:
```bash
python src/speech_to_text.py  --model vosk-model-small-en-us-0.15
```

### Google Cloud Speech-to-Text

Experimenting with cloud STT, seems of decent quality and speed.

To run the Google Cloud speech-to-text example:
```bash
python src/google_cloud_speech_cli.py
```

## Talking to the companion

The companion runs a wake word recognition server that launches speech recognition when activated and uses the Gemini streaming API to communicate. It streams the recognized words to Gemini and streams the AI's responses back to you in real-time.

To run the AI companion:
```bash
python src/talk_to_ai.py --wake_keyword bumblebee
```

The companion will:
1. Listen for the wake word ("bumblebee" by default)
2. When the wake word is detected, start listening for your command
3. Convert your speech to text
4. Send your command to Gemini AI
5. Stream Gemini's response back to you
6. Continue to listen for further commands until you stop the program

## License and Attribution

This project uses Picovoice's Porcupine library, which requires a valid access key. See [Picovoice's licensing terms](https://picovoice.ai/docs/terms-of-use/) for more information.