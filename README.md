# LeAI Companion

An AI companion for Leia that uses wake word detection to enable voice-activated interactions.

## Overview

LeAI Companion uses Picovoice's Porcupine wake word detection library to listen for specific keywords, allowing for hands-free activation of your AI companion.

## Prerequisites

- Python 3.7 or later
- A microphone connected to your computer
- Picovoice Access Key (get it from [Picovoice Console](https://console.picovoice.ai/))

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

4. Create a `.env` file in the project root and add your Picovoice access key:
   ```
   PICOVOICE_ACCESS_KEY=your_access_key_here
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
   python src/main.py
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

## License and Attribution

This project uses Picovoice's Porcupine library, which requires a valid access key. See [Picovoice's licensing terms](https://picovoice.ai/docs/terms-of-use/) for more information.