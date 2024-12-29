import os
import csv
import whisper

from datasets import load_dataset


def transcribe_audio_files(directory: str, output_csv: str):
    """
    Iterates through the given directory, transcribes any audio files
    using the Whisper ASR model, and stores ALL results in a single CSV.

    :param directory: Path to the folder containing audio files.
    :param output_csv: Path (including filename) for the output CSV file.
    """

    # Load the Whisper model (options: tiny, base, small, medium, large)
    model = whisper.load_model("base")

    # Define valid audio extensions
    audio_extensions = (".wav", ".mp3", ".m4a", ".flac", ".ogg")

    # A list to gather (file_name, transcript) for all audio files
    results = []

    # Iterate over files in the directory
    for file_name in os.listdir(directory):
        if file_name.lower().endswith(audio_extensions):
            audio_path = os.path.join(directory, file_name)
            print(f"Transcribing: {audio_path}")

            # Transcribe the audio file
            result = model.transcribe(audio_path)
            transcript = result["text"]

            # Add the result to our list
            results.append((file_name, transcript))

    # Write all results into a single CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["file_name", "transcript", "file"])
        # Write each row of data
        for file_name, transcript in results:
            base_name = os.path.basename(file_name)
            writer.writerow([file_name, transcript, base_name])

    print(f"Done! Results saved to '{output_csv}'.")


if __name__ == "__main__":
    # Change these paths to your desired directory and output CSV
    # directory = r"D:\Documents\MASC\MSP_POD_dataset\Audios\Audios.tar\test"
    # output_csv = r"D:\Documents\MASC\MSP_POD_dataset\Audios\Audios.tar\test\metadata.csv"

    # transcribe_audio_files(directory, output_csv)\\
    dirname = r"D:\Documents\MASC\MSP_POD_dataset\Audios\Audios.tar\test"

    dataset = load_dataset("audiofolder", data_dir=dirname)
    dataset.push_to_hub("MSPP_WAV_test")
