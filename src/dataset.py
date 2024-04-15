import os
import sys
import librosa
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset
from tqdm import tqdm
from glob import glob
import pandas as pd
from .constants import *

class SARAGA_CARNATIC(Dataset):
    def __init__(self, path, hop_length, sequence_length=None, groups=None, sample_rate=CARNATIC_SAMPLE_RATE, window_length=WINDOW_LENGTH, freq_hop_length=CARNATIC_FREQ_HOP_LENGTH):
        self.path = path
        self.SAMPLE_RATE = sample_rate
        self.WINDOW_LENGTH = window_length
        self.HOP_LENGTH = int(hop_length / 1000 * self.SAMPLE_RATE)
        self.SEQ_LEN = None if not sequence_length else int(sequence_length * self.SAMPLE_RATE)
        self.num_class = N_CLASS
        self.data = []
        self.FREQ_HOP_LENGTH = int(freq_hop_length / 1000 * self.SAMPLE_RATE)

        print(f"Loading {len(groups)} group{'s' if len(groups) > 1 else ''} "
              f"of {self.__class__.__name__} at {path}")
        for group in groups:
            for input_files in tqdm(self.files(group), desc='Loading group %s' % group):
                self.data.extend(self.load(*input_files))

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def availabe_groups():
        return ['test']
    
    def files(self, group):
        audio_files = glob(os.path.join(self.path, group, '*.wav'))
        label_files = [f.replace('.wav', '.pv') for f in audio_files]

        assert (all(os.path.isfile(audio_v_file) for audio_v_file in audio_files))
        assert (all(os.path.isfile(label_file) for label_file in label_files))

        return sorted(zip(audio_files, label_files))

    def load(self, audio_path, label_path):
        data = []
        audio, _ = librosa.load(audio_path, sr=self.SAMPLE_RATE)
        audio_l = len(audio)

        audio = np.pad(audio, self.WINDOW_LENGTH // 2, mode='reflect')
        audio = torch.from_numpy(audio).float()

        audio_steps = audio_l // self.HOP_LENGTH + 1
        freq_per_hop = self.HOP_LENGTH // self.FREQ_HOP_LENGTH
        print(f"Audio_l: {audio_l}, self.HOP_LENGTH: {self.HOP_LENGTH}, audio_steps: {audio_steps}, freq_per_hop: {freq_per_hop}", file=sys.stderr)

        # Pitch label: processed F0 value. A matrix of audio_steps height and 360 width (the classes)
        pitch_label = torch.zeros(audio_steps, self.num_class, dtype=torch.float)
        # Voice label: whether there is a voice at the segment
        voice_label = torch.zeros(audio_steps, dtype=torch.float)
        with open(label_path, 'r') as f:
            lines = f.readlines()
            current_sum = 0
            count = 0
            j = 0
            for line in lines:
                # We take the average freq in every hop. SARAGA dataset has a freq hop length
                # of 4ms, which is too fine for use.
                current_sum += float(line)
                count += 1

                if count == freq_per_hop:
                    avg_freq = current_sum / freq_per_hop
                    if avg_freq != 0:
                        cent = 1200 * np.log2(avg_freq/BASE_CENT_FREQ)
                        index = int(round((cent-CONST)/20))
                        # print(f"Average freq for hop {j} of file {label_path}: {avg_freq} (index: {index})", file=sys.stderr)
                        if j < audio_steps:
                            try:
                                pitch_label[j][index] = 1
                                voice_label[j] = 1
                            except IndexError:
                                print(f"IndexError with freq {avg_freq}, index {index}, j {j} in file {label_path}", file=sys.stderr)
                                sys.exit(1)
                    j += 1
                    current_sum = 0
                    count = 0
        
        if self.SEQ_LEN is not None:
            segment_length = self.SEQ_LEN + self.WINDOW_LENGTH
            step_size = self.SEQ_LEN
            num_segments = (len(audio) - self.WINDOW_LENGTH) // step_size
            for i in range(num_segments + 1):
                begin_t = i * step_size
                end_t = begin_t + segment_length
                if end_t > len(audio):
                    end_t = len(audio)
                    begin_t = max(0, end_t - segment_length)
                segment = audio[begin_t:end_t]
                if len(segment) < segment_length:
                    segment = torch.nn.functional.pad(segment, (0, segment_length - len(segment)), 'constant', 0)
                
                begin_step = begin_t // self.HOP_LENGTH
                end_step = min((begin_t + segment_length) // self.HOP_LENGTH, len(pitch_label))
                pitch_segment = pitch_label[begin_step:end_step]
                voice_segment = voice_label[begin_step:end_step]
                expected_length = segment_length // self.HOP_LENGTH

                missing_frames = expected_length - len(pitch_segment)
                if missing_frames > 0:
                    # Pad the missing frames at the end of the sequence dimension
                    pitch_segment = F.pad(pitch_segment, (0, 0, 0, missing_frames), 'constant', 0)
                    voice_segment = F.pad(voice_segment, (0, missing_frames), 'constant', 0)
                
                print(f"expected_length: {expected_length}, file: {audio_path}_segment{i}, audio_segment dimension: {len(segment)}, pitch_segment dimension: {len(pitch_segment)}x{len(pitch_segment[0])}, voice_segment len: {len(voice_segment)}", file=sys.stderr)

                data.append({
                    'audio': segment,
                    'pitch': pitch_segment,
                    'voice': voice_segment,
                    'file': audio_path
                })
        else:
            data.append({
                'audio': audio,
                'pitch': pitch_label,
                'voice': voice_label,
                'file': audio_path
            })

        return data

class MIR1K(Dataset):
    def __init__(self, path, hop_length, sequence_length=None, groups=None):
        self.path = path
        self.HOP_LENGTH = int(hop_length / 1000 * SAMPLE_RATE)
        self.seq_len = None if not sequence_length else int(sequence_length * SAMPLE_RATE)
        self.num_class = N_CLASS
        self.data = []

        print(f"Loading {len(groups)} group{'s' if len(groups) > 1 else ''} "
              f"of {self.__class__.__name__} at {path}")
        for group in groups:
            for input_files in tqdm(self.files(group), desc='Loading group %s' % group):
                self.data.extend(self.load(*input_files))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def availabe_groups():
        return ['test']

    def files(self, group):
        audio_files = glob(os.path.join(self.path, group, '*.wav'))
        label_files = [f.replace('.wav', '.pv') for f in audio_files]

        assert (all(os.path.isfile(audio_v_file) for audio_v_file in audio_files))
        assert (all(os.path.isfile(label_file) for label_file in label_files))

        return sorted(zip(audio_files, label_files))

    def load(self, audio_path, label_path):
        data = []
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
        audio_l = len(audio)

        audio = np.pad(audio, WINDOW_LENGTH // 2, mode='reflect')
        audio = torch.from_numpy(audio).float()

        audio_steps = audio_l // self.HOP_LENGTH + 1

        pitch_label = torch.zeros(audio_steps, self.num_class, dtype=torch.float)
        voice_label = torch.zeros(audio_steps, dtype=torch.float)
        with open(label_path, 'r') as f:
            lines = f.readlines()
            i = 0
            for line in lines:
                i += 1
                if float(line) != 0:
                    freq = 440 * (2.0 ** ((float(line) - 69.0) / 12.0))
                    cent = 1200 * np.log2(freq/10)
                    index = int(round((cent-CONST)/20))
                    pitch_label[i][index] = 1
                    voice_label[i] = 1

        if self.seq_len is not None:
            n_steps = self.seq_len // self.HOP_LENGTH + 1
            for i in range(audio_l // self.seq_len):
                begin_t = i * self.seq_len
                end_t = begin_t + self.seq_len + WINDOW_LENGTH
                begin_step = begin_t // self.HOP_LENGTH
                end_step = begin_step + n_steps
                data.append(dict(audio=audio[begin_t:end_t], pitch=pitch_label[begin_step:end_step],
                                 voice=voice_label[begin_step:end_step], file=audio_path))
            data.append(dict(audio=audio[-self.seq_len - WINDOW_LENGTH:], pitch=pitch_label[-n_steps:],
                             voice=voice_label[-n_steps:], file=audio_path))
        else:
            data.append(dict(audio=audio, pitch=pitch_label, voice=voice_label, file=audio_path))
        return data


class MIR_ST500(Dataset):
    def __init__(self, path, hop_length, sequence_length=None, groups=None):
        self.path = path
        self.HOP_LENGTH = int(hop_length / 1000 * SAMPLE_RATE)
        self.seq_len = None if not sequence_length else int(sequence_length * SAMPLE_RATE)
        self.num_class = N_CLASS
        self.data = []

        print(f"Loading {len(groups)} group{'s' if len(groups) > 1 else ''} "
              f"of {self.__class__.__name__} at {path}")
        for group in groups:
            for input_files in tqdm(self.files(group), desc='Loading group %s' % group):
                self.data.extend(self.load(*input_files))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def availabe_groups():
        return ['test']

    def files(self, group):
        audio_files = glob(os.path.join(self.path, group, '*.wav'))
        label_files = [f.replace('.wav', '.tsv') for f in audio_files]

        assert (all(os.path.isfile(audio_v_file) for audio_v_file in audio_files))
        assert (all(os.path.isfile(label_file) for label_file in label_files))

        return sorted(zip(audio_files, label_files))

    def load(self, audio_path, label_path):
        data = []
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
        audio_l = len(audio)

        audio = np.pad(audio, WINDOW_LENGTH // 2, mode='reflect')
        audio = torch.from_numpy(audio).float()

        audio_steps = audio_l // self.HOP_LENGTH + 1

        pitch_label = torch.zeros(audio_steps, self.num_class, dtype=torch.float)
        voice_label = torch.zeros(audio_steps, dtype=torch.float)

        midi = np.loadtxt(label_path, delimiter='\t', skiprows=1)
        for onset, offset, note in midi:
            left = int(round(onset * SAMPLE_RATE / self.HOP_LENGTH))
            right = int(round(offset * SAMPLE_RATE / self.HOP_LENGTH)) + 1
            freq = 440 * (2.0 ** ((float(note) - 69.0) / 12.0))
            cent = 1200 * np.log2(freq / 10)
            index = int(round((cent - CONST) / 20))
            pitch_label[left:right, index] = 1
            voice_label[left:right] = 1

        if self.seq_len is not None:
            n_steps = self.seq_len // self.HOP_LENGTH + 1
            for i in range(audio_l // self.seq_len):
                begin_t = i * self.seq_len
                end_t = begin_t + self.seq_len + WINDOW_LENGTH
                begin_step = begin_t // self.HOP_LENGTH
                end_step = begin_step + n_steps
                data.append(dict(audio=audio[begin_t:end_t], pitch=pitch_label[begin_step:end_step],
                                 voice=voice_label[begin_step:end_step], file=audio_path))
            data.append(dict(audio=audio[-self.seq_len - WINDOW_LENGTH:], pitch=pitch_label[-n_steps:],
                             voice=voice_label[-n_steps:], file=audio_path))
        else:
            data.append(dict(audio=audio, pitch=pitch_label, voice=voice_label, file=audio_path))
        return data


class MDB(Dataset):
    def __init__(self, path, hop_length, sequence_length=None, groups=None):
        self.path = path
        self.HOP_LENGTH = int(hop_length / 1000 * SAMPLE_RATE)
        self.seq_len = None if not sequence_length else int(sequence_length * SAMPLE_RATE)
        self.num_class = N_CLASS
        self.data = []

        print(f"Loading {len(groups)} group{'s' if len(groups) > 1 else ''} "
              f"of {self.__class__.__name__} at {path}")
        for group in groups:
            for input_files in tqdm(self.files(group), desc='Loading group %s' % group):
                self.data.extend(self.load(*input_files))

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

    @staticmethod
    def availabe_groups():
        return ['test']

    def files(self, group):
        audio_files = glob(os.path.join(self.path, group, '*.wav'))
        label_files = [f.replace('.wav', '.csv') for f in audio_files]

        assert (all(os.path.isfile(audio_v_file) for audio_v_file in audio_files))
        assert (all(os.path.isfile(label_file) for label_file in label_files))

        return sorted(zip(audio_files, label_files))

    def load(self, audio_path, label_path):
        data = []
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
        audio_l = len(audio)

        audio = np.pad(audio, WINDOW_LENGTH // 2, mode='reflect')
        audio = torch.from_numpy(audio).float()

        audio_steps = audio_l // self.HOP_LENGTH + 1

        pitch_label = torch.zeros(audio_steps, self.num_class, dtype=torch.float)
        voice_label = torch.zeros(audio_steps, dtype=torch.float)

        df_label = pd.read_csv(label_path)
        for i in range(len(df_label)):
            if float(df_label['midi'][i]):
                freq = 440 * (2.0 ** ((float(df_label['midi'][i]) - 69.0) / 12.0))
                cent = 1200 * np.log2(freq / 10)
                index = int(round((cent - CONST) / 20))
                pitch_label[i][index] = 1
                voice_label[i] = 1

        if self.seq_len is not None:
            n_steps = self.seq_len // self.HOP_LENGTH + 1
            for i in range(audio_l // self.seq_len):
                begin_t = i * self.seq_len
                end_t = begin_t + self.seq_len + WINDOW_LENGTH
                begin_step = begin_t // self.HOP_LENGTH
                end_step = begin_step + n_steps
                data.append(dict(audio=audio[begin_t:end_t], pitch=pitch_label[begin_step:end_step],
                                 voice=voice_label[begin_step:end_step], file=audio_path))
            data.append(dict(audio=audio[-self.seq_len - WINDOW_LENGTH:], pitch=pitch_label[-n_steps:],
                             voice=voice_label[-n_steps:], file=audio_path))
        else:
            data.append(dict(audio=audio, pitch=pitch_label, voice=voice_label, file=audio_path))
        return data
