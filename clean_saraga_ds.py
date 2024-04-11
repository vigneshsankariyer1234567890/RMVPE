import logging
import os
import random
import shutil
import sys
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count, Pool
import librosa
import numpy as np
import soundfile as sf
from glob import glob
from scipy.io import wavfile
from typing import List, Tuple

COLLECTION_PATH = 'collections'
COLLECTION_2_PATH = 'collections_2'
MP3_MP3_SUFFIX = '.mp3.mp3'
MP3_SUFFIX = '.mp3'
PITCH_TXT_SUFFIX = '.pitch.txt'
WAV_SUFFIX = '.wav'
PV_SUFFIX = '.pv'

TEST_SET_PATH = 'test'
TRAIN_SET_PATH = 'train'
TEST_PROB = 0.8

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_collections_directory(collections_dir: str) -> List[Tuple[str, str]]:
    wav_files = glob(os.path.join(collections_dir, '*' + WAV_SUFFIX))
    pv_files = [f.replace(WAV_SUFFIX, PV_SUFFIX) for f in wav_files]
    return zip(wav_files, pv_files)

def create_directory(base_path: str, directory_name: str) -> str:
    full_path = os.path.join(base_path, directory_name)

    if not os.path.exists(full_path):
        os.makedirs(full_path)
        print(f"Directory created: {full_path}\n")
    else:
        print(f"Directory already exists: {full_path}\n")
    
    return full_path

def check_files_exist(file_paths):
    return all(os.path.isfile(file_path) for file_path in file_paths)

def process_song_directory(full_path: str) -> List[Tuple[str, str]]:
    pair_list = []
    audio_mp3_mp3_files = glob(os.path.join(full_path, '*' + MP3_MP3_SUFFIX))
    label_mp3_mp3_files = [f.replace(MP3_MP3_SUFFIX, PITCH_TXT_SUFFIX) for f in audio_mp3_mp3_files]

    if len(audio_mp3_mp3_files) == len(label_mp3_mp3_files) and \
       check_files_exist(audio_mp3_mp3_files) and check_files_exist(label_mp3_mp3_files):
        pair_list.extend(zip(audio_mp3_mp3_files, label_mp3_mp3_files))
    
    audio_mp3_files = glob(os.path.join(full_path, '*' + MP3_SUFFIX))
    label_mp3_files = [f.replace(MP3_SUFFIX, PITCH_TXT_SUFFIX) for f in audio_mp3_files]

    if len(audio_mp3_files) == len(label_mp3_files) and \
       check_files_exist(audio_mp3_files) and check_files_exist(label_mp3_files):
        pair_list.extend(zip(audio_mp3_files, label_mp3_files))

    for audio, pitch in pair_list:
        print(f"Audio: {audio}, Pitch: {pitch}")
    
    return pair_list

def collect_mp3_pitch_pairs(base_path: str) -> List[Tuple[str, str]]:
    pair_list = []
    directories = []

    for concert in os.listdir(base_path):
        concert_path = os.path.join(base_path, concert)
        if os.path.isdir(concert_path):
            for song in os.listdir(concert_path):
                full_path = os.path.join(concert_path, song)
                if os.path.isdir(full_path):
                    directories.append(full_path)

    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(process_song_directory, dir_path) for dir_path in directories]
        for future in as_completed(futures):
            pair_list.extend(future.result())

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
    
    print(f"Conversion completed and written to {pv_file_path}\n")

def process_file_pair(audio: str, pitch: str, collection_dir: str, dataset_dir: str):
    truncated_audio_path = audio[len(dataset_dir) + 1:].replace('/', '_')
    truncated_pitch_path = pitch[len(dataset_dir) + 1:].replace('/', '_')

    if truncated_audio_path.endswith(MP3_MP3_SUFFIX):
        trunc_audio_path_as_wav = remove_suffix_and_append(truncated_audio_path, MP3_MP3_SUFFIX, WAV_SUFFIX)
    else:
        trunc_audio_path_as_wav = remove_suffix_and_append(truncated_audio_path, MP3_SUFFIX, WAV_SUFFIX)
    trunc_audio_path_as_wav_with_collection = os.path.join(collection_dir, trunc_audio_path_as_wav)
    trunc_pitch_path_as_pv = remove_suffix_and_append(truncated_pitch_path, PITCH_TXT_SUFFIX, PV_SUFFIX)
    trunc_pitch_path_as_pv_with_collection = os.path.join(collection_dir, trunc_pitch_path_as_pv)

    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(convert_mp3_to_wav, audio, trunc_audio_path_as_wav_with_collection)
        executor.submit(process_pitchtxt_to_pv, pitch, trunc_pitch_path_as_pv_with_collection)

    return (trunc_audio_path_as_wav_with_collection, trunc_pitch_path_as_pv_with_collection)

def copy_to_collection_dir(collection_dir: str, audio_label_pairs: List[Tuple[str, str]], dataset_dir: str) -> List[Tuple[str]]:
    with Pool(cpu_count()) as pool:
        results = pool.starmap(process_file_pair, [(audio, pitch, collection_dir, dataset_dir) for audio, pitch in audio_label_pairs])
    return results

def get_file_name(file_path: str) -> str:
    filename_with_extension = os.path.basename(file_path)
    filename_without_extension, _ = os.path.splitext(filename_with_extension)
    return filename_without_extension

def get_formatted_file_name(prefix: str, file_name: str, identifier: str, extension: str) -> str:
    return os.path.join(prefix, f'{file_name}_{identifier}{extension}')

def segment_audio(audio_path: str, segment_size: float, push_to_dir: str) -> List[str]:
    filename_without_extension = get_file_name(audio_path)
    
    sample_rate, data = wavfile.read(audio_path)
    frames_per_segment = int(segment_size * sample_rate)
    segmented_audio_paths = []

    num_segments = (len(data) + frames_per_segment - 1) // frames_per_segment

    for segment_idx in range(num_segments):
        start = segment_idx * frames_per_segment
        end = start + frames_per_segment
        segment_data = data[start:end]
        segment_audio_path = get_formatted_file_name(push_to_dir, filename_without_extension, str(segment_idx), WAV_SUFFIX)

        wavfile.write(segment_audio_path, sample_rate, segment_data)

        segmented_audio_paths.append(segment_audio_path)
    
    return segmented_audio_paths

def segment_pitch(pitch_path: str, segment_size: float, push_to_dir: str) -> List[str]:
    filename_without_extension = get_file_name(pitch_path)

    segmented_pitch_paths = []
    start_time = 0.0
    segment_index = 0

    with open(pitch_path, 'r') as file:
        lines = file.readlines()
    
    while True:
        segment_pitch_path = get_formatted_file_name(push_to_dir, filename_without_extension, str(segment_index), PV_SUFFIX)
        segment_has_data = False

        with open(segment_pitch_path, 'w') as spf:
            for line in lines:
                time_str, freq_str = line.strip().split()
                time = float(time_str)
                frequency = float(freq_str)
                if start_time <= time < start_time + segment_size:
                    spf.write(f"{frequency}\n")
                    if not segment_has_data:
                      segment_has_data = True
                elif time >= start_time + segment_size:
                    break
        
        if not segment_has_data:
            os.remove(segment_pitch_path)
            break
        
        segmented_pitch_paths.append(segment_pitch_path)
        start_time += segment_size
        segment_index += 1
    
    return segmented_pitch_paths    

def segment_audio_pitch_pair(audio_pitch_pair: Tuple[str, str], segment_size: float, push_to_dir: str) -> List[Tuple[str, str]]:
    audio, pitch = audio_pitch_pair

    dir_path = os.path.dirname(audio)
    parent_dir = os.path.dirname(dir_path)
    new_audio_path = os.path.join(parent_dir, push_to_dir, get_file_name(audio))
    detected_existing_files = glob(f'{new_audio_path}_*{WAV_SUFFIX}')
    if len(detected_existing_files) > 0:
        return []

    new_audio_files = segment_audio(audio, segment_size, push_to_dir)
    new_pitch_files = segment_pitch(pitch, segment_size, push_to_dir)

    return list(zip(new_audio_files, new_pitch_files))

def split_collections_data(file_pairs:List[Tuple[str, str]], test_prob: float, test_dir: str, train_dir: str):
    for audio, pitch in file_pairs:
        if not check_files_exist([audio]) or not check_files_exist([pitch]):
            continue
        if random.random() > test_prob:
            shutil.move(audio, os.path.join(test_dir, os.path.basename(audio)))
            shutil.move(pitch, os.path.join(test_dir, os.path.basename(pitch)))
        else:
            shutil.move(audio, os.path.join(train_dir, os.path.basename(audio)))
            shutil.move(pitch, os.path.join(train_dir, os.path.basename(pitch)))

def remove_directory(dir_path: str):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)

def main(dataset_dir: str):
    collection_dir = create_directory(dataset_dir, COLLECTION_PATH)

    audio_label_pairs = collect_mp3_pitch_pairs(dataset_dir)

    new_audio_label_pairs = copy_to_collection_dir(collection_dir, audio_label_pairs, dataset_dir)

    collection_2_dir = create_directory(dataset_dir, COLLECTION_2_PATH)

    new_audio_list = []
    for audio_pitch_pair in new_audio_label_pairs:
        segmented_list = segment_audio_pitch_pair(audio_pitch_pair, 10.0, collection_2_dir)
        new_audio_list.extend(segmented_list)

    test_dir = create_directory(dataset_dir, TEST_SET_PATH)
    train_dir = create_directory(dataset_dir, TRAIN_SET_PATH)
    split_collections_data(new_audio_list, TEST_PROB, test_dir, train_dir)
    remove_directory(collection_dir)
    remove_directory(collection_2_dir)

def copy_to_collection(dataset_dir: str):
    collection_dir = create_directory(dataset_dir, COLLECTION_PATH)
    audio_label_pairs = collect_mp3_pitch_pairs(dataset_dir)
    copy_to_collection_dir(collection_dir, audio_label_pairs, dataset_dir)

def segment_audio_pitch(dataset_dir: str):
    collection_dir = create_directory(dataset_dir, COLLECTION_PATH)
    new_audio_label_pairs = read_collections_directory(collection_dir)
    collection_2_dir = create_directory(dataset_dir, COLLECTION_2_PATH)

    with Pool(cpu_count()) as pool:
        pool.starmap(segment_audio_pitch_pair, [(pair, 10.0, collection_2_dir) for pair in new_audio_label_pairs])

def separate_into_test_train(dataset_dir: str):
    collection_dir = create_directory(dataset_dir, COLLECTION_PATH)
    collection_2_dir = create_directory(dataset_dir, COLLECTION_2_PATH)
    new_audio_label_pairs = read_collections_directory(collection_2_dir)

    test_dir = create_directory(dataset_dir, TEST_SET_PATH)
    train_dir = create_directory(dataset_dir, TRAIN_SET_PATH)
    split_collections_data(new_audio_label_pairs, TEST_PROB, test_dir, train_dir)
    remove_directory(collection_dir)
    remove_directory(collection_2_dir)

def clean_test_train_directory(dir: str, ext_to_remove: str, ext_to_check: str):
    files = glob(os.path.join(dir, '*' + ext_to_remove))
    for file in files:
        if not check_files_exist([os.path.join(dir, f'{get_file_name(file)}{ext_to_check}')]):
            os.remove(file)

def verify_integrity(dir: str):
    wav_files = glob(os.path.join(dir, '*' + WAV_SUFFIX))
    for file in wav_files:
        if not check_files_exist([os.path.join(dir, f'{get_file_name(file)}{PV_SUFFIX}')]):
            print(f'No corresponding PV file for {file}')

    pitch_files = glob(os.path.join(dir, '*' + PV_SUFFIX))
    for file in pitch_files:
        if not check_files_exist([os.path.join(dir, f'{get_file_name(file)}{WAV_SUFFIX}')]):
            print(f'No corresponding WAV file for {file}')

def pitch_pairs_collection_test(dataset_dir: str):
    collection_dir = create_directory(dataset_dir, COLLECTION_PATH)

    start_time = time.time()
    pairs = collect_mp3_pitch_pairs(dataset_dir)
    new_audio_label_pairs = copy_to_collection_dir(collection_dir, pairs, dataset_dir)
    elapsed_time = time.time() - start_time
    logging.info(f"collect_mp3_pitch_pairs executed in {elapsed_time:.2f} seconds.\n")
    logging.info(f"collected {len(pairs)} pairs of audio files\n")
    logging.info(f"created {len(new_audio_label_pairs)} new pairs.\n")

    remove_directory(collection_dir)

def integration_test(dataset_dir: str):
    collection_dir = create_directory(dataset_dir, COLLECTION_PATH)

    test_list = [('/home/svu/e0552366/e0552366/saraga1.5_carnatic/Akkarai Sisters at Arkay by Akkarai Sisters/Apparama Bhakti/Apparama Bhakti.mp3.mp3', '/home/svu/e0552366/e0552366/saraga1.5_carnatic/Akkarai Sisters at Arkay by Akkarai Sisters/Apparama Bhakti/Apparama Bhakti.pitch.txt'), ('/home/svu/e0552366/e0552366/saraga1.5_carnatic/Ashwath Narayanan at Arkay by Ashwath Narayanan/Angakaram/Ashwath Narayanan - Angakaram.mp3', '/home/svu/e0552366/e0552366/saraga1.5_carnatic/Ashwath Narayanan at Arkay by Ashwath Narayanan/Angakaram/Ashwath Narayanan - Angakaram.pitch.txt')]

    new_audio_label_pairs = copy_to_collection_dir(collection_dir, test_list, dataset_dir)

    collection_2_dir = create_directory(dataset_dir, COLLECTION_2_PATH)
    
    new_audio_list = []
    for audio_pitch_pair in new_audio_label_pairs:
        segmented_list = segment_audio_pitch_pair(audio_pitch_pair, 10.0, collection_2_dir)
        new_audio_list.extend(segmented_list)
    test_dir = create_directory(dataset_dir, TEST_SET_PATH)
    train_dir = create_directory(dataset_dir, TRAIN_SET_PATH)

    split_collections_data(new_audio_list, TEST_PROB, test_dir, train_dir)

    remove_directory(collection_dir)
    remove_directory(collection_2_dir)
    assert(os.path.exists(test_dir) and os.path.isdir(test_dir))
    print(f'Number of test files = {len(os.listdir(test_dir))}\n')
    assert(os.path.exists(train_dir) and os.path.isdir(train_dir))
    print(f'Number of train files = {len(os.listdir(train_dir))}\n')
    remove_directory(test_dir)
    remove_directory(train_dir)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python clean_ds.py <dataset_dir>")
        sys.exit(1)

    dataset_dir = sys.argv[1]
    separate_into_test_train(dataset_dir)
    # segment_audio_pitch(dataset_dir)
    # copy_to_collection(dataset_dir)
    # main(dataset_dir)
    # pitch_pairs_collection_test(dataset_dir)
    # integration_test(dataset_dir)
