# Dependencies
from tqdm.notebook import tqdm_notebook
import csv
from PIL import Image
from tqdm import tqdm
import soundfile as sf
import parselmouth
from pydub import AudioSegment
import numpy as np
import pandas as pd
import os
import librosa
import matplotlib.pyplot as plt
import gc
import time
from tqdm import tqdm, tqdm_notebook
tqdm.pandas()  # Progress bar
# from sklearn.metrics import label_ranking_average_precision_score
# from sklearn.model_selection import train_test_split


t_start = time.time()
# import noisereduce as nr

Datasets = '/media/carol/Data/DATASETS/Emotion Datasets'


# (a) MFCC (b) Chromagram (c) Spectral Contrast (d) Mel spectrogram (e) Tonnetz.


def clean_audio(path):
    y, sr = librosa.load(path)
    S_full, phase = librosa.magphase(librosa.stft(y))
    idx = slice(*librosa.time_to_frames([2, 6], sr=sr))
    width = int((S_full.shape[-1] - 1)/2)-1
    S_filter = librosa.decompose.nn_filter(S_full,
                                           aggregate=np.median,
                                           metric='cosine',
                                           width=width)
    S_filter = np.minimum(S_full, S_filter)
    margin_i, margin_v = 2, 10
    power = 2

    mask_i = librosa.util.softmask(S_filter,
                                   margin_i * (S_full - S_filter),
                                   power=power)

    mask_v = librosa.util.softmask(S_full - S_filter,
                                   margin_v * S_filter,
                                   power=power)

    S_foreground = mask_v * S_full

    sound = librosa.istft(S_foreground * phase)
    # sound = y
    # # # sf.write(os.path.join(new_dir,new_name), librosa.istft(S_foreground * phase), sr)
    return sound, sr


def draw_spectrogram(ax, spectrogram, dynamic_range=70):
    X, Y = spectrogram.x_grid(), spectrogram.y_grid()
    eps = 1e-10
    sg_db = 10 * np.log10(spectrogram.values+eps)
    img = ax.pcolormesh(X, Y, sg_db, vmin=sg_db.max() -
                        dynamic_range, cmap='Spectral', shading='auto')
    ax.set_ylim([spectrogram.ymin, spectrogram.ymax])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_frame_on(False)  # Turn off the frame (white border)
    return img


def draw_intensity(ax, intensity):
    # Plot only the thin black line
    ax.plot(intensity.xs(), intensity.values.T, linewidth=1, color='#FFA500')
    # Fill the area under the curve
    ax.fill_between(intensity.xs(), 0, intensity.values.T.flatten(),
                    alpha=0.5, color='#FFA500')  # Light orange
    # Set y-axis limit
    ax.set_ylim(0)
    # Remove grid
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_frame_on(False)  # Turn off the frame (white border)

    return ax


def draw_mfcc(ax, mfcc, dynamic_range=70):
    # X = mfcc.centre_time
    # Y = np.arange(mfcc.n_frames)
    sg_db = mfcc.to_array()
    # You can adjust the constant if needed
    sg_db = 10 * np.log10(sg_db + 1e-10)
    img = ax.pcolormesh(sg_db, vmin=sg_db.max() -
                        dynamic_range, cmap='Spectral', shading='auto')
    # ax.set_ylim([0, mfcc.n_coefficients])
    # ax.set_xlim([mfcc.start_time, mfcc.end_time])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_frame_on(False)  # Turn off the frame (white border)
    return img


def draw_lib(ax, lib):
    img = ax.pcolormesh(lib, cmap='Spectral', shading='auto')
    # ax.set_ylim([0, mfcc.n_coefficients])
    # ax.set_xlim([mfcc.start_time, mfcc.end_time])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_frame_on(False)  # Turn off the frame (white border)
    return img


def make_img(x, sr):
    snd = parselmouth.Sound(x)
    S = np.abs(librosa.stft(x, n_fft=512))**2
    chroma = librosa.feature.chroma_stft(S=S, sr=sr)
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    harm = librosa.effects.harmonic(x)
    tonnetz = librosa.feature.tonnetz(y=harm, sr=sr)
    # Compute the Mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(
        y=x, sr=sr, n_mels=128, n_fft=512, hop_length=256, fmax=sr/2
    )

    # Convert power to decibel scale for better visualization
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Assuming snd, to_intensity, to_spectrogram, to_pitch, and pre_emphasize are defined
    mfccs = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40)
    intensity = snd.to_intensity()

    spectrogram = snd.to_spectrogram()

    pitch = snd.to_pitch()
    pre_emphasized_snd = snd.copy()
    pre_emphasized_snd.pre_emphasize()
    spectrogram_preemphasized = pre_emphasized_snd.to_spectrogram(
        window_length=0.08, maximum_frequency=8000)
    plt.show()
    # Create a compressed 2x1 subplot grid with desired figsize
    # Adjust the figsize as needed
    fig, axes = plt.subplots(3, 2, figsize=(224/100, 224/100))

    draw_lib(axes[0, 0], mel_spectrogram_db)
    draw_spectrogram(axes[0, 1], spectrogram)

    # Plot the first spectrogram on the top subplot
    # draw_spectrogram(ax, spectrogram)
    # draw_spectrogram(axes[0, 0], spectrogram)

    # draw_spectrogram(axes[0, 1], spectrogram_preemphasized)

    draw_lib(axes[1, 0], mfccs)
    draw_lib(axes[1, 1], chroma)
    draw_lib(axes[2, 0], contrast)
    draw_lib(axes[2, 1], tonnetz)

    # Adjust spacing between subplots
    plt.subplots_adjust(hspace=0)

    # Convert the figure to a numpy array
    fig.tight_layout(pad=0)

    # Drawing the canvas
    fig.canvas.draw()
    plt.close(fig)

    # Getting the pixel data without the border
    data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (4,))

    plt.close()

    return data


def process_files(df, image_dir, audio_dir, csv_dir, ds=0):
    os.makedirs(image_dir, exist_ok=True)
    csv_file = os.path.join(csv_dir, 'file_labels.csv')

    # Keep track of existing files
    existing_files = set()
    if os.path.exists(csv_file):
        with open(csv_file, 'r', encoding='utf-8', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                existing_files.add(row['file_name'])

    j = 0

    with open(csv_file, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['file_name', 'file', 'transcript']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if os.path.getsize(csv_file) == 0:
            writer.writeheader()

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing audio files"):
            file_path = os.path.join(audio_dir, row['file_name'])
            file_name = row['file_name']
            label_val = row['file_name']
            speaker = row['transcript']
            if ds == 0:
                fname = str(file_name.split('.wav')[0])

            else:
                indval = file_path.rfind('/', 0, file_path.rfind('/'))
                vid_utter = file_path[indval+1:]
                fname = vid_utter.replace("/", "_")

            ''' IF OMG DATASET NEED TO CHHANGE TO THIS
                        
            '''

            jpname = str(fname + '.png')
            img_path = os.path.join(fname + '.png')  # Save as PNG
            # labelled_dir = os.path.join(image_dir, str(label_val))
            # os.makedirs(labelled_dir) if not os.path.exists(labelled_dir) else None

            full_path = os.path.join(image_dir, img_path)
            if img_path in existing_files:
                print(f"Skipping {file_name} as image already exists.")
                continue
            else:
                j += 1
            # if j>10000:
            #     break

            x, sr = clean_audio(file_path)
            s = parselmouth.Sound(x)
            if s.get_total_duration() < 2 * 0.09:  # Assuming window_length is 0.08 seconds
                print(f"Skipping {file_name} due to short duration.")
                continue
            data = make_img(x, sr)
            img = Image.fromarray(data)
            img.save(full_path)
            # print('image saved to', full_path)
            writer.writerow(
                {'file_name': jpname, 'file': label_val, 'transcript': speaker})


# Display available datasets
print("Available datasets:")
print("1. CMU MOSEI")
print("2. CREMA-D")
print("3. EMOV-DB")
print("4. MSP IMPROV")
print("5. RAVDESS-DB")
print("6. TESS-DB")
print("7. VIVAE-DB")
print("8. IEMOCAP-DB")
print("9. ASVP-ESD")
print("10. OMG")
print("11. MSP_POD")
# print("12. CUSTOM3")


# Get user input for dataset choice
# dataset_choice = int(
#     input("Enter the number of the dataset you would like to run: "))
dataset_choice = 2

if dataset_choice == 1:

    # CMU MOSEI
    print('--------------EMOV-DB DATASET STARTED ---- ')

    file_path = r'D:\Documents\MASC\EMOV_DB\New_Folders\metadata.csv'

    df = pd.read_csv(file_path)

    image_dir = r'D:\Documents\MASC\EMOV_DB\Image_DS'
    os.makedirs(image_dir, exist_ok=True)
    csv_dir = r'D:\Documents\MASC\EMOV_DB\Image_DS\metadata.csv'
    os.makedirs(csv_dir, exist_ok=True)
    process_files(df, image_dir, csv_dir)

    #################################################################################


elif dataset_choice == 2:
    print('--------------EMOV DATASET STARTED ---- ')

    file_path = r'D:\Documents\MASC\EMOV_DB\New_Folders\metadata.csv'
    df = pd.read_csv(file_path)
    audio_dir = r"D:\Documents\MASC\EMOV_DB\New_Folders"
    image_dir = r'D:\Documents\MASC\EMOV_DB\Image_DS'
    os.makedirs(image_dir, exist_ok=True)
    csv_dir = r'D:\Documents\MASC\EMOV_DB\Image_DS'
    os.makedirs(csv_dir, exist_ok=True)
    process_files(df, image_dir, audio_dir, csv_dir, 0)


elif dataset_choice == 11:
    print('--------------MSP_POD DATASET STARTED ---- ')

    file_path = r'D:\Documents\MASC\MSP_POD_dataset\Audios\Audios.tar\Audios\metadata.csv'
    df = pd.read_csv(file_path)
    audio_dir = r"D:\Documents\MASC\MSP_POD_dataset\Audios\Audios.tar\Audios"
    image_dir = r'D:\Documents\MASC\MSP_POD_dataset\Image_DS\Old_Mel'
    os.makedirs(image_dir, exist_ok=True)
    csv_dir = r'D:\Documents\MASC\MSP_POD_dataset\Image_DS\Old_Mel'
    os.makedirs(csv_dir, exist_ok=True)
    process_files(df, image_dir, audio_dir, csv_dir, 0)


elif dataset_choice == 12:

    print('--------------MSP_POD TEST DATASET STARTED ---- ')

    file_path = r'D:\Documents\MASC\MSP_POD_dataset\Audios\Audios.tar\test\metadata.csv'
    df = pd.read_csv(file_path)
    audio_dir = r"D:\Documents\MASC\MSP_POD_dataset\Audios\Audios.tar\test"
    image_dir = r'D:\Documents\MASC\MSP_POD_dataset\Image_DS\TEST_oldfeats'
    os.makedirs(image_dir, exist_ok=True)
    csv_dir = r'D:\Documents\MASC\MSP_POD_dataset\Image_DS\TEST_oldfeats'
    os.makedirs(csv_dir, exist_ok=True)
    process_files(df, image_dir, audio_dir, csv_dir, 0)
