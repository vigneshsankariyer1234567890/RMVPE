import os
import librosa
import numpy as np
import torch

from torch.utils.data import Dataset
from tqdm import tqdm
from glob import glob
import pandas as pd
from .constants import *

class SARAGA_CARNATIC(Dataset):
    def __init__(self, path, hop_length, sequence_length=None, groups=None, sample_rate=CARNATIC_SAMPLE_RATE, window_length=WINDOW_LENGTH):
        self.path = path
        self.SAMPLE_RATE = sample_rate
        self.WINDOW_LENGTH = window_length
        self.HOP_LENGTH = int(hop_length / 1000 * self.SAMPLE_RATE)
        self.SEQ_LEN = None if not sequence_length else int(sequence_length * self.SAMPLE_RATE)
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
        audio, _ = librosa.load(audio_path, sr=self.SAMPLE_RATE)
        audio_l = len(audio)

        audio = np.pad(audio, self.WINDOW_LENGTH // 2, mode='reflect')
        audio = torch.from_numpy(audio).float()

        audio_steps = audio_l // self.HOP_LENGTH + 1

        # Pitch label: processed F0 value. A matrix of audio_steps height and 360 width (the classes)
        pitch_label = torch.zeros(audio_steps, self.num_class, dtype=torch.float)
        # Voice label: whether there is a voice at the segment
        voice_label = torch.zeros(audio_steps, dtype=torch.float)
        with open(label_path, 'r') as f:
            lines = f.readlines()
            i = 0
            for line in lines:
                i += 1
                if float(line) != 0:
                    # Get the frequency from the pitch label
                    freq =float(line)
                    # Get the cent value (against BASE_CENT_FREQ)
                    cent = 1200 * np.log2(freq/BASE_CENT_FREQ)
                    # Normalize cent and get the right bucket
                    index = int(round((cent-CONST)/20))
                    # Mark bucket in segment of pitch as 1
                    pitch_label[i][index] = 1
                    # Mark segment in voice as 1
                    voice_label[i] = 1
        
        if self.SEQ_LEN is not None:
            n_steps = self.SEQ_LEN // self.HOP_LENGTH + 1
            for i in range(audio_l // self.SEQ_LEN):
                begin_t = i * self.SEQ_LEN
                end_t = begin_t + self.SEQ_LEN + self.WINDOW_LENGTH
                begin_step = begin_t // self.HOP_LENGTH
                end_step = begin_step + n_steps
                data.append(dict(
                    audio=audio[begin_t:end_t], 
                    pitch=pitch_label[begin_step:end_step],
                    voice=voice_label[begin_step:end_step],
                    file=audio_path
                ))

            data.append(dict(
                audio=audio[-self.SEQ_LEN - self.WINDOW_LENGTH:],
                pitch=pitch_label[-n_steps:],
                voice=voice_label[-n_steps:],
                file=audio_path
            ))
        else:
            data.append(dict(
                audio=audio,
                pitch=pitch_label,
                voice=voice_label,
                file=audio_path
            ))
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
