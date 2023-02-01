import argparse

from transcriber import WhisperTranscriber


def main() -> None:
    # Parse arguments from the command-line.
    command_args = parse_arguments()

    transcriber = WhisperTranscriber(**vars(command_args))

    # Start the transcriber!
    print("Transcriber is ready! Recording in progress.")
    transcriber.start()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", default="medium", help="Model to use.",
                        choices=["tiny", "base", "small", "medium", "medium.en", "large"])
    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"],
                        help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')")
    parser.add_argument("--energy_threshold", default=700,
                        help="Energy level for microphone to detect.", type=int)
    parser.add_argument("--phrase_timeout", default=2,
                        help="How long it will wait to start processing new phrases. This "
                             "effectively controls the responsiveness of the transcriber.",
                        type=float)
    parser.add_argument("--sentence_timeout", default=3,
                        help="How long it will wait after the last phrase to treat it as the end "
                             "of the sentence.",
                        type=float)
    parser.add_argument("--language", type=str, default=None,
                        help="The main language spoken in the speech, specify None to auto-detect or support multiple languages.")
    parser.add_argument("--languages", nargs="+", default=["en"],
                        help="List of languages to use for transcribing.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
