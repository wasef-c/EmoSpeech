import os
import shutil


def copy_test_audio_files(
    source_dir: str,
    destination_dir: str,
    audio_extensions=None,
    case_sensitive: bool = False
):
    """
    Copies audio files from source_dir to destination_dir
    if they contain the word 'test' in their filename.

    :param source_dir: Path to the directory containing audio files.
    :param destination_dir: Path to the directory where files should be copied.
    :param audio_extensions: List or set of acceptable audio file extensions (e.g. ['.mp3', '.wav']).
    :param case_sensitive: If False, 'test' is matched case-insensitively.
    """

    if audio_extensions is None:
        # Default to a few common audio formats if none provided
        audio_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg'}

    # # Make sure destination directory exists
    # if not os.path.exists(destination_dir):
    #     os.makedirs(destination_dir, exist_ok=True)

    # Traverse the source directory
    for filename in os.listdir(source_dir):
        file_path = os.path.join(source_dir, filename)

        # Skip if it's not a file
        if not os.path.isfile(file_path):
            continue

        # Extract file extension
        _, ext = os.path.splitext(filename)

        # Check if it's an audio file
        # if ext.lower() in audio_extensions:
        #     # Check if filename contains "test"
        #     if case_sensitive:
        #         if "test" in filename:
        #             shutil.copy2(file_path, destination_dir)
        #             print(f"Copied: {filename}")
        #     else:
        #         if "test" in filename.lower():
        #             shutil.copy2(file_path, destination_dir)
        #             print(f"Copied: {filename}")
        if ext.lower() in audio_extensions:
            # Check if filename contains "test"
            if case_sensitive:
                if "test" in filename:
                    os.remove(file_path)
                    print(f"Deleted: {filename}")
            else:
                if "test" in filename.lower():
                    os.remove(file_path)
                    print(f"Deleted: {filename}")


if __name__ == "__main__":
    # Replace these paths with your actual directories
    source = r"D:\Documents\MASC\MSP_POD_dataset\Audios\Audios.tar\Audios"
    destination = r"D:\Documents\MASC\MSP_POD_dataset\Audios\Audios.tar\Audios\test"

    copy_test_audio_files(source, destination, case_sensitive=False)
