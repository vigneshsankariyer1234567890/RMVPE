import os
import sys

import librosa
import soundfile as sf
from glob import glob
from typing import List, Tuple

COLLECTION_PATH = 'collections'
MP3_SUFFIX = '.mp3.mp3'
PITCH_TXT_SUFFIX = '.pitch.txt'
WAV_SUFFIX = '.wav'
PV_SUFFIX = '.pv'

def create_directory(base_path: str, directory_name: str) -> str:
    full_path = os.path.join(base_path, directory_name)

    if not os.path.exists(full_path):
        os.makedirs(full_path)
        print(f"Directory created: {full_path}")
    else:
        print(f"Directory already exists: {full_path}")
    
    return full_path

def check_files_exist(file_paths):
    return all(os.path.isfile(file_path) for file_path in file_paths)

def collect_mp3_pitch_pairs(base_path: str) -> List[Tuple[str, str]]:
    pair_list = []

    for concert in os.listdir(base_path):
        if os.path.isfile(os.path.join(base_path, concert)):
            continue
        for song in os.listdir(os.path.join(base_path, concert)):
            full_path = os.path.join(base_path, concert, song)

            audio_files = glob(os.path.join(full_path, '*' + MP3_SUFFIX))
            label_files = [f.replace(MP3_SUFFIX, PITCH_TXT_SUFFIX) for f in audio_files]

            if len(audio_files) == len(label_files) and \
                check_files_exist(audio_files) and \
                check_files_exist(label_files):
                for pair in zip(audio_files, label_files):
                    pair_list.append(pair)
    return pair_list

def remove_suffix_and_append(file_path: str, suffix_to_remove: str, suffix_to_append: str) -> str:
    if file_path.endswith(suffix_to_remove):
        new_length = len(file_path) - len(suffix_to_remove)
        root = file_path[:new_length]
    else:
        root = file_path
    
    new_file_path = root + suffix_to_append

    return new_file_path

def convert_mp3_to_wav(mp3_file_path: str, wav_file_path: str):
    audio, sample_rate = librosa.load(mp3_file_path, sr=None)

    sf.write(wav_file_path, audio, sample_rate)
    print(f"Converted {mp3_file_path} to {wav_file_path} successfully.\n")

def process_pitchtxt_to_pv(pitchtxt_file_path: str, pv_file_path: str):
    with open(pitchtxt_file_path, 'r') as file:
        lines = file.readlines()
    
    with open(pv_file_path, 'w') as output_file:
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            
            time_str, freq_str = parts[0], parts[1]

            time = float(time_str)
            frequency = float(freq_str)
            formatted_time = f"{time:.7f}"
            formatted_frequency = f"{frequency:.7f}"

            output_file.write(f"{formatted_time}\t{formatted_frequency}\n")
    
    print(f"Conversion completed and written to {pv_file_path}")

def copy_to_collection_dir(collection_dir: str, audio_label_pairs: List[Tuple[str, str]], dataset_dir: str) -> List[Tuple[str]]:
    new_audio_label_pairs = []
    for audio, pitch in audio_label_pairs:
        if not (audio.startswith(dataset_dir) and pitch.startswith(dataset_dir)):
            raise ValueError("Paths do not start with the dataset directory.")

        truncated_audio_path = audio[len(dataset_dir) + 1:].replace('/', '_')
        truncated_pitch_path = pitch[len(dataset_dir) + 1:].replace('/', '_')

        trunc_audio_path_as_wav = remove_suffix_and_append(truncated_audio_path, MP3_SUFFIX, WAV_SUFFIX)
        trunc_audio_path_as_wav_with_collection = os.path.join(collection_dir, trunc_audio_path_as_wav)
        convert_mp3_to_wav(audio, trunc_audio_path_as_wav_with_collection)

        trunc_pitch_path_as_pv = remove_suffix_and_append(truncated_pitch_path, PITCH_TXT_SUFFIX, PV_SUFFIX)
        trunc_pitch_path_as_pv_with_collection = os.path.join(collection_dir, trunc_pitch_path_as_pv)
        process_pitchtxt_to_pv(pitch, trunc_pitch_path_as_pv_with_collection)

        new_audio_label_pairs.append((trunc_audio_path_as_wav_with_collection, trunc_pitch_path_as_pv_with_collection))
    
    return new_audio_label_pairs


def main(dataset_dir: str):
    collection_dir = create_directory(dataset_dir, COLLECTION_PATH)

    # audio_label_pairs = collect_mp3_pitch_pairs(dataset_dir)

    # new_audio_label_pairs = copy_to_collection_dir(collection_dir, audio_label_pairs, dataset_dir)

    test_list = [('/home/svu/e0552366/e0552366/saraga1.5_carnatic/Akkarai Sisters at Arkay by Akkarai Sisters/Apparama Bhakti/Apparama Bhakti.mp3.mp3', '/home/svu/e0552366/e0552366/saraga1.5_carnatic/Akkarai Sisters at Arkay by Akkarai Sisters/Apparama Bhakti/Apparama Bhakti.pitch.txt')]

    new_audio_label_pairs = copy_to_collection_dir(collection_dir, test_list, dataset_dir)

    for audio, pitch in new_audio_label_pairs:
        print(f'New audio: {audio}\nNew pitch: {pitch}\n')
        

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python clean_ds.py <dataset_dir>")
        sys.exit(1)

    dataset_dir = sys.argv[1]
    main(dataset_dir)



