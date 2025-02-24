import os
import csv
import whisper
import pandas as pd


def transcribe_audio_files(directory: str, metadata_csv: str):
    """
    Reads an existing metadata.csv, adds a 'transcript' column with Whisper-generated transcriptions,
    and saves the updated file.

    :param directory: Path to the folder containing audio files.
    :param metadata_csv: Path to the existing metadata CSV file.
    """

    # Load the Whisper model (options: tiny, base, small, medium, large)
    model = whisper.load_model("base")

    # Define valid audio extensions
    audio_extensions = (".wav", ".mp3", ".m4a", ".flac", ".ogg")

    # Load existing metadata.csv
    if os.path.exists(metadata_csv):
        metadata_df = pd.read_csv(metadata_csv)
    else:
        print(f"Error: '{metadata_csv}' does not exist!")
        return

    # Ensure 'file_name' column exists in metadata
    if "file_name" not in metadata_df.columns:
        print("Error: 'file_name' column not found in metadata.csv!")
        return

    # Iterate over files in the directory
    transcripts = {}
    for file_name in os.listdir(directory):
        if file_name.lower().endswith(audio_extensions):
            audio_path = os.path.join(directory, file_name)
            print(f"Transcribing: {audio_path}")

            # Transcribe the audio file
            result = model.transcribe(audio_path)
            transcripts[file_name] = result["text"]

    # Add transcript column to metadata_df (only for matching file_name entries)
    metadata_df["transcript"] = metadata_df["file_name"].map(transcripts)

    # Save the updated CSV
    metadata_df.to_csv(metadata_csv, index=False)
    print(f"Updated metadata saved to '{metadata_csv}'.")


if __name__ == "__main__":
    # Change these paths to your desired directory and metadata CSV file
    directory = r"D:\Documents\MASC\EMOV_DB\New_Folders"
    metadata_csv = r"D:\Documents\MASC\EMOV_DB\New_Folders\metadata.csv"

    transcribe_audio_files(directory, metadata_csv)
