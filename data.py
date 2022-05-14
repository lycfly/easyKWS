import torch
import numpy as np
from torch.utils.data import Dataset
import librosa
from glob import glob
import random
from skimage.transform import resize
import pandas as pd
from random import sample
import os
import torchaudio
import sys
from quantize_audio_feature import qFeature, log10fx

SR=16000

class SpeechDataset(Dataset):
    def __init__(self, noise_path, mode, label_words_dict, wav_list, add_noise, preprocess_fun, feature_cfg = {}, sr=SR, resize_shape=None, is_1d=False, ppq_enable=False):
        """Args:
                mode: train or evaluate or test
                label_words_dict: a dict of words for labels
                wav_list: a list of wav file paths
                add_noise: boolean. if background noise should be added
                preprocess_fun: function to load/process wav file
                preprocess_param: params for preprocess_fun
                sr: default 16000
                resize_shape: None. only for 2d cnn.
                is_1d: boolean. if it is going to be 1d cnn or 2d cnn
        """
        self.mode = mode
        self.label_words_dict = label_words_dict
        self.wav_list = wav_list
        self.add_noise = add_noise
        self.sr = sr
        self.n_silence = int(len(wav_list) * 0.09)
        self.preprocess_fun = preprocess_fun
        self.feature_cfg = feature_cfg
        self.qlist =  feature_cfg['qlist']
        self.ppq_enable = ppq_enable
        # read all background noise here
        self.noise_path = noise_path
        self.background_noises = [librosa.load(x, sr=self.sr)[0] for x in glob(os.path.join(self.noise_path,"*.wav"))]
        self.resize_shape = resize_shape
        self.is_1d = is_1d
        self.use_gpu = self.feature_cfg["feature_use_gpu"]
        self.use_quantize = self.feature_cfg["use_quantize"]

        if self.use_quantize:
            self.qf = qFeature(qlist = self.qlist, feature_cfg = self.feature_cfg)
            self.feature_cfg['qf'] = self.qf

    def get_one_noise(self):
        """generates one single noise clip"""
        selected_noise = self.background_noises[random.randint(0, len(self.background_noises) - 1)]
        # only takes out 16000
        start_idx = random.randint(0, len(selected_noise) - 1 - self.sr)
        return selected_noise[start_idx:(start_idx + self.sr)]

    def get_mix_noises(self, num_noise=1, max_ratio=0.1):
        result = np.zeros(self.sr)
        for _ in range(num_noise):
            result += random.random() * max_ratio * self.get_one_noise()
        return result / num_noise if num_noise > 0 else result

    def get_one_word_wav(self, idx):
        wav = librosa.load(self.wav_list[idx], sr=self.sr)[0]
        if len(wav) < self.sr:
            padding_len = self.sr - len(wav)
            wav = np.pad(wav, (0, padding_len), 'constant')
        return wav[:self.sr]

    def get_silent_wav(self, num_noise=1, max_ratio=0.5):
        return self.get_mix_noises(num_noise=num_noise, max_ratio=max_ratio)

    def timeshift(self, wav, ms=100):
        shift = (self.sr * ms) // 1000
        shift = random.randint(-shift, shift)
        a = -min(0, shift)
        b = max(0, shift)
        data = np.pad(wav, (a, b), "constant")
        return data[:len(data) - a] if a else data[b:]

    def get_noisy_wav(self, idx):
        scale = random.uniform(0.75, 1.25)
        num_noise = random.choice([1, 2])
        max_ratio = random.choice([0.1, 0.5, 1, 1.5])
        mix_noise_proba = random.choice([0.1, 0.3])
        shift_range = random.randint(80, 120)
        one_word_wav = self.get_one_word_wav(idx)
        if random.random() < mix_noise_proba:
            return scale * (self.timeshift(one_word_wav, shift_range) + self.get_mix_noises(
                num_noise, max_ratio))
        else:
            return one_word_wav

    def __len__(self):
        if self.mode == 'test':
            return len(self.wav_list)
        else:
            return len(self.wav_list) + self.n_silence

    def __getitem__(self, idx):
        """reads one sample"""
        if idx < len(self.wav_list):
            wav_numpy = self.preprocess_fun(
                self.get_one_word_wav(idx) if self.mode != 'train' else self.get_noisy_wav(idx),
                **self.feature_cfg)
            if self.resize_shape:
                wav_numpy = resize(wav_numpy, (self.resize_shape, self.resize_shape), preserve_range=True)
            if self.use_gpu:
                wav_tensor = wav_numpy
            else:
                wav_tensor = torch.from_numpy(wav_numpy).float()
            if not self.is_1d:
                wav_tensor = wav_tensor.unsqueeze(0)
            #if self.mode == 'test':
            #    return {'spec': wav_tensor, 'id': self.wav_list[idx]}

            label = self.label_words_dict[self.wav_list[idx].split("/")[-2]] if self.wav_list[idx].split(
                "/")[-2] in self.label_words_dict else len(self.label_words_dict)
            if self.ppq_enable:
                return wav_tensor
            else:
                return {'spec': wav_tensor, 'id': self.wav_list[idx], 'label': label}

        else:
            """generates silence here"""
            wav_numpy = self.preprocess_fun(self.get_silent_wav(
                num_noise=random.choice([0, 1, 2, 3]),
                max_ratio=random.choice([x / 10. for x in range(20)])), **self.feature_cfg)
            if self.resize_shape:
                wav_numpy = resize(wav_numpy, (self.resize_shape, self.resize_shape), preserve_range=True)
            if self.use_gpu:
                wav_tensor = wav_numpy
            else:
                wav_tensor = torch.from_numpy(wav_numpy).float()
            if not self.is_1d:
                wav_tensor = wav_tensor.unsqueeze(0)
            if self.ppq_enable:
                return wav_tensor
            else:
                return {'spec': wav_tensor, 'id': 'silence', 'label': len(self.label_words_dict) + 1}


def get_label_dict(words):
    label_to_int = dict(zip(words, range(len(words))))
    int_to_label = dict(zip(range(len(words)), words))
    int_to_label.update({len(words): 'unknown', len(words) + 1: 'silence'})
    return label_to_int, int_to_label


def get_wav_list(path, target_dataset, words, unknown_ratio=0.2):
    if target_dataset == 'GSDC_v01' or target_dataset == 'GSDC_v02':
        full_train_list = glob(os.path.join(path,"train/*/*.wav"))
        full_val_list   = glob(os.path.join(path,"valid/*/*.wav"))
        full_test_list  = glob(os.path.join(path,"test/*/*.wav"))

        # sample full train list
        sampled_train_list = []
        for w in full_train_list:
            l = w.split("/")[-2]
            if l not in words:
                if random.random() < unknown_ratio:
                    sampled_train_list.append(w)
            else:
                sampled_train_list.append(w)
    else:
        raise('There is no this dataset!')
    return sampled_train_list, full_val_list, full_test_list



def get_semi_list(words, sub_path, unknown_ratio=0.2, test_ratio=0.2):
    train_list, _ = get_wav_list(words=words, unknown_ratio=unknown_ratio)
    test_list = get_sub_list(num=int(len(train_list) * test_ratio), sub_path=sub_path)
    lst = train_list + test_list
    return sample(lst, len(lst))


def preprocess_mfcc(wave, sr=SR, n_mels=40, hop_length=160, n_fft=480, fmin=20, fmax=4000):
    spectrogram = librosa.feature.mfcc(wave, sr=sr, S=None, n_mfcc=n_mels, n_mels=n_mels, hop_length=hop_length, 
                                     n_fft=n_fft, fmin=fmin, fmax=fmax)
    mfcc = mfcc.astype(np.float32)
    return mfcc


def preprocess_mel(data, sample_rate=SR, n_mels=40, n_fft=480, win_length=480, hop_length=160,
                           normalization=False, feature_use_gpu=True):
    if feature_use_gpu:
        window = torch.hann_window
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate = sample_rate,
            n_fft = n_fft,
            win_length = win_length,
            hop_length = hop_length,
            center= True,
            pad_mode = "reflect",
            n_mels = n_mels,
            power = 2.0,
            f_min = 20,
            f_max = 4000,
            window_fn = window
        )
        mel_sepctrogram = mel_spectrogram.cuda()
        spectrogram = mel_sepctrogram(torch.tensor(data).cuda().to(torch.float))
        spectrogram = torchcaudio.functional.amplitude_to_DB(spectrogram, multiplier=20.0, amin=1e-10, db_multiplier=0.0, top_db=80)
    else:    
        spectrogram = librosa.feature.melspectrogram(data, sr=SR, n_mels=n_mels, hop_length=160, n_fft=480, fmin=20, fmax=4000)
        spectrogram = librosa.power_to_db(spectrogram)
        spectrogram = spectrogram.astype(np.float32)
    if normalization:
        spectrogram = spectrogram.spectrogram()
        spectrogram -= spectrogram
    return spectrogram

def preprocess_mel_my(data, sr=SR, n_mels=40, n_fft=480, win_length=480, hop_length=160, fmin=20, fmax=4000, qf=None,
                           normalization=False, feature_use_gpu=True, use_quantize=False, qlist=None):
    if feature_use_gpu:
        window = torch.hann_window
        flip_num = int(win_length/2)
        mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate = sr,
            n_fft = flip_num,
            win_length = flip_num,
            hop_length = flip_num,
            center= False,
            pad_mode = "reflect",
            n_mels = n_mels,
            power = 2.0,
            f_min = fmin,
            f_max = fmax,
            window_fn = window
        )
        mel_sepctrogram = mel_spectrogram.cuda()
        spectrogram = mel_sepctrogram(torch.tensor(data).cuda().to(torch.float))
        spectrogram = (spectrogram[:,:-1] + spectrogram[:,1:])
        spectrogram = torchcaudio.functional.amplitude_to_DB(spectrogram, multiplier=20.0, amin=1e-10, db_multiplier=0.0, top_db=80)
    
    elif use_quantize:
        spec = qf.fbank_quantize(data,db=False)
        spectrogram = (spec[:,:-1] + spec[:,1:])
        spectrogram = qf.cordic_ln.ln(spectrogram) * 8
        spectrogram = spectrogram.astype(np.float32)
    else:    
        spectrogram = librosa.feature.melspectrogram(data, sr=sr, n_mels=n_mels, hop_length=hop_length, n_fft=n_fft, fmin=fmin, fmax=fmax)
        spectrogram = (spectrogram[:,:-1] + spectrogram[:,1:])
        spectrogram = librosa.power_to_db(spectrogram)
        spectrogram = spectrogram.astype(np.float32)
    if normalization:
        spectrogram = spectrogram.spectrogram()
        spectrogram -= spectrogram
    return spectrogram
    
def preprocess_wav(wav, normalization=True):
    data = wav.reshape(1, -1)
    if normalization:
        mean = data.mean()
        data -= mean
    return data