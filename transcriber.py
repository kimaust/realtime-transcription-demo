from time import sleep
from queue import Queue
from typing import Optional
from tempfile import NamedTemporaryFile
from datetime import datetime, timedelta

import io
import whisper
import speech_recognition as sr


class WhisperTranscriber:
    def __init__(self, **kwargs) -> None:
        super().__init__()

        self._kwargs = kwargs

        self._load_model(self._kwargs["model"])
        self._setup_speech_recognizer()
        self._setup_audio_source()

        # Create thread-safe queue to store the raw bytes representing a phrase we want to
        # transcribe. Detected phrases (in raw bytes) are added via `__phrase_callback`` function.
        self._phrase_data_queue = Queue()
        self._last_audio_sample = bytes()

        self._last_transcription_text = ""
        self._last_phrase_time = None
        self._has_completed_sentence = False
        self._last_sentence_completion_time = None

        # Since whisper requires a file to transcribe, get a temporary filename to store the wave audio data.
        self._audio_source_filename = NamedTemporaryFile().name

    def start(self) -> None:
        # Create a background thread to listen for the audio from the microphone.
        self.recognizer.listen_in_background(
            self.audio_source,
            self._phrase_callback,
            phrase_time_limit=self._kwargs["phrase_timeout"])

        while True:
            elapsed_time = self._get_elapsed_time(self._last_phrase_time)

            # If enough time has elapsed since the last phrase time, consider it as the end of a sentence.
            self._has_completed_sentence = elapsed_time >= timedelta(seconds=self._kwargs["sentence_timeout"])
            self._handle_phrases()

            # Sleep for a bit to prevent this loop from hogging the CPU.
            sleep(0.25)

    def _load_model(self, model_name: str) -> None:
        self.model = whisper.load_model(model_name)

    def _setup_speech_recognizer(self) -> None:
        # Use SpeechRecognizer to record our audio. Also turn off dynamic energy threshold to
        # prevent the SpeechRecognizer from lowering the energy threshold automatically.
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = self._kwargs["energy_threshold"]
        self.recognizer.dynamic_energy_threshold = False

    def _setup_audio_source(self) -> None:
        self.sample_width = 2
        self.sample_rate = 48000
        self.audio_source = sr.Microphone(sample_rate=self.sample_rate)

    def _phrase_callback(self, _, phrase_data: sr.AudioData) -> None:
        """A callback function that is called when the SpeechRecognizer detects a phrase."""
        # Get the raw bytes from the audio data and push it into the thread safe queue.
        self._phrase_data_queue.put(phrase_data.get_raw_data())

    def _handle_phrases(self):
        if self._phrase_data_queue.qsize() == 0:
            punctuations = (".", "..", "...", "?", "!")
            if self._last_transcription_text.endswith(punctuations):
                return

            # If there are no new phrases to process and we have completed a sentence, add a full-stop.
            valid_completed_sentence = self._last_transcription_text and self._has_completed_sentence
            if valid_completed_sentence:
                text_with_fullstop = f"{self._last_transcription_text}."
                self._last_transcription_text = text_with_fullstop
                print(text_with_fullstop)
        else:
            # Update the last phrase time as there are new phrases.
            self._last_phrase_time = datetime.utcnow()
            self._save_audio_from_sample()

            transcription_result = self.model.transcribe(
                self._audio_source_filename,
                task=self._kwargs["task"],
                language=self._kwargs["language"],
                languages=self._kwargs["languages"])
            new_transcription_text = transcription_result['text'].strip()

            # Only process the transcription text if it isn't empty as whisper sometimes produces empty results.
            if not new_transcription_text:
                return
            # If new phrases are detected, but results in the same trasncription within the sentence, ignore it.
            if new_transcription_text == self._last_transcription_text and not self._has_completed_sentence:
                return

            print(new_transcription_text)
            self._last_transcription_text = new_transcription_text

    def _save_audio_from_sample(self) -> None:
        # If we've completed sentence, clear the current working audio buffer to start over with the new data.
        if self._has_completed_sentence:
            self._last_audio_sample = bytes()
            self._last_sentence_completion_time = datetime.utcnow()

        # Pull all the audio data from the queue and concatenate it into a single audio sample.
        while not self._phrase_data_queue.empty():
            data = self._phrase_data_queue.get()
            self._last_audio_sample += data

        # Use AudioData to convert the raw audio data to wave data.
        audio_data = sr.AudioData(
            self._last_audio_sample,
            self.sample_rate,
            self.sample_width)
        wav_data = io.BytesIO(audio_data.get_wav_data())

        # Write wav data to the temporary file as bytes for whisper to transcribe.
        with open(self._audio_source_filename, "w+b") as file:
            file.write(wav_data.read())

    @staticmethod
    def _get_elapsed_time(last_time: Optional[datetime] = None) -> timedelta:
        if last_time is None:
            return timedelta(seconds=0)

        return datetime.utcnow() - last_time
