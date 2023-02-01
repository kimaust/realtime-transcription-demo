# Realtime Speech-to-Text Transcription Demo
A simple CLI realtime speech-to-text transcription using Whisper.

This uses the idea from the [whisper_real_time](https://github.com/studentofkyoto/whisper_real_time) to record audio in a background thread and concatenating the raw bytes over multiple recordings.

Also, my fork of whisper [whisper-lang-selection](https://github.com/kimaust/whisper-lang-selection) is used to support a selection of languages to detect.

# Setup
Create a virtual environment using the following command:
```bash
python -m venv venv
```

And activate the virtual environment:
```bash
source venv/bin/activate # for linux
./venv/Scripts/Activate  # for Windows
```

Finally, install the required packages using:

```bash
pip install -r requirements.txt
```

Note that you will also need to install [ffmpeg](https://ffmpeg.org) to be installed on your system.

# Demo
You can run the realtime demo using the following command:
```bash
python transcribe.py
```