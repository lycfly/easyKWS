import numpy as np
import os
import librosa
import matplotlib.pyplot as plt
import scipy
from librosa._cache import cache
import librosa.util as util
from FFT import *
import time


@cache(level=20)
def my_get_window(window, Nx, fftbins=True):
    return scipy.signal.get_window(window, Nx, fftbins=fftbins)


@cache(level=20)
def get_qfft_window(
        n_fft=2048,
        window="hann",
        win_length=None,
        qlist=(1, 8, 7),
):
    s, w, f = qlist
    fft_window = librosa.filters.get_window(window, win_length, fftbins=True)
    fft_window = librosa.util.pad_center(fft_window, n_fft)
    # Reshape so that the window can be broadcast
    fft_window = fft_window.reshape((-1, 1))
    fft_window = np.array(numfi(fft_window, s, w, f))
    return fft_window


@cache(level=20)
def frame(
        y,
        n_fft=2048,
        hop_length=None,
        center=False,
        pad_mode="reflect",
):
    # Check audio is valid
    if (center):
        y = np.pad(y, int(n_fft // 2), mode=pad_mode)
    # Window the time series.
    y_frames = util.frame(y, frame_length=n_fft, hop_length=hop_length)
    return y_frames

@cache(level=20)
def frame_window(
        y,
        n_fft=2048,
        hop_length=None,
        center=False,
        pad_mode="reflect",
        window = None,
):
    # Check audio is valid
    if (center):
        y = np.pad(y, int(n_fft // 2), mode=pad_mode)
    # Window the time series.
    y_frames = util.frame(y, frame_length=n_fft, hop_length=hop_length)
    y_frames = window * y_frames
    return y_frames

@cache(level=20)
def my_stft(
        y_frames,
        n_fft=2048,
        dtype=None,
        fft_window=None,
        fft_handle=None,
):
    if dtype is None:
        dtype = util.dtype_r2c(y_frames.dtype)
    stft_matrix = np.empty(
        (int(1 + n_fft // 2), y_frames.shape[1]), dtype=dtype, order="F"
    )
    # fft = librosa.get_fftlib()
    fft = fft_handle
    n_columns = util.MAX_MEM_BLOCK // (stft_matrix.shape[0] * stft_matrix.itemsize)
    n_columns = max(n_columns, 1)

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])

        stft_matrix[:, bl_s:bl_t] = fft(
            fft_window * y_frames[:, bl_s:bl_t], rfft=True
        )
        # stft_matrix[:, bl_s:bl_t] = fft.rfft(
        #     fft_window * y_frames[:, bl_s:bl_t], axis=0
        # )
    return stft_matrix

@cache(level=20)
def my_stft_nowindow(
        y_frames,
        n_fft=2048,
        dtype=None,
        fft_handle=None,
):
    if dtype is None:
        dtype = util.dtype_r2c(y_frames.dtype)
    stft_matrix = np.empty(
        (int(1 + n_fft // 2), y_frames.shape[1]), dtype=dtype, order="F"
    )
    # fft = librosa.get_fftlib()
    fft = fft_handle
    n_columns = util.MAX_MEM_BLOCK // (stft_matrix.shape[0] * stft_matrix.itemsize)
    n_columns = max(n_columns, 1)

    for bl_s in range(0, stft_matrix.shape[1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[1])

        stft_matrix[:, bl_s:bl_t]  = fft(
            y_frames[:, bl_s:bl_t], rfft=True
        )
        # stft_matrix[:, bl_s:bl_t] = fft.rfft(
        #     fft_window * y_frames[:, bl_s:bl_t], axis=0
        # )
    return stft_matrix

def abs_power(fft_matrix):
    return fft_matrix.real ** 2 + fft_matrix.imag ** 2


def abs_add(fft_matrix):
    return np.abs(fft_matrix.real) + np.abs(fft_matrix.imag)


def quantized_mel_filter(sr=16000, n_fft=256, n_mels=40, fmin=20, fmax=4000, qlist=(1, 16, 15)):
    s, w, f = qlist
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax)
    mel_basis_q = np.array(numfi(mel_basis * 64, s, w, f))
    return mel_basis_q


def spectrum(mel_basis, mel_basis_q):
    melspec = np.dot(mel_basis, mel_basis_q)
    return melspec


def logfx_config(amin, points, ofmt, tanfmt):
    xlist = np.array([amin * (2 ** x) for x in range(points)])
    fx = np.log10
    ylist = 10 * fx(xlist)
    if amin == 0:
        ylist[0] = - 2 ** (ofmt[1] - ofmt[0] - ofmt[2] - 1)
    tan_list = (np.array(ylist)[1:] - np.array(ylist)[:-1]) / (np.array(xlist)[1:] - np.array(xlist)[:-1])
    y_qlist = np.array(numfi(ylist, ofmt[0], ofmt[1], ofmt[2]))
    tan_qlist = np.array(numfi(tan_list, tanfmt[0], tanfmt[1], tanfmt[2]))
    return xlist, y_qlist, tan_qlist


def log10fx(indata, xlist, y_qlist, tan_qlist, amin, points, ofmt):
    indata = (indata == 0) * amin + indata
    index = (np.log2(indata / amin)).astype(np.int32)
    index = np.where(index > points - 2, points - 2, index)
    out = y_qlist[index] + (indata - xlist[index]) * tan_qlist[index]
    out = np.array(numfi(out, ofmt[0], ofmt[1], ofmt[2]))

    return out


class FixedCordic():
    def __init__(self, qinput=None, qcoeff=(0, 8, 8), iter_index=np.arange(16)):
        self.s_in, self.w_in, self.f_in = qinput
        self.s_coef, self.w_coef, self.f_coef = qcoeff
        self.iter_index = iter_index
        self.iter_num = np.array(iter_index).shape[0] + 1
        self.width_add = int(np.log(self.iter_num) / np.log(2))

        self.mode_dict = {'circle': 0, 'liner': 1, 'hyperbolic': 2}

        e1 = np.arctan(np.power(0.5, iter_index))
        e2 = np.power(0.5, iter_index)
        ## for log
        neg_e = np.array(iter_index)[np.where(np.array(iter_index) <= 0)]
        e3_p1 = np.arctanh(1 - np.power(0.5, -neg_e + 2))
        pos_e = np.array(iter_index)[np.where(np.array(iter_index) > 0)]
        e3_p2 = np.arctanh(np.power(0.5, pos_e))
        e3 = np.concatenate((e3_p1, e3_p2))
        e3_fixed = numfi(e3, self.s_coef, self.w_coef, self.f_coef)
        e2_fixed = numfi(e2, self.s_coef, self.w_coef, self.f_coef)
        e1_fixed = numfi(e1, self.s_coef, self.w_coef, self.f_coef)

        self.e = {'circle': e1_fixed, 'liner': e2_fixed, 'hyperbolic': e3_fixed}
        self.u = {'circle': numfi(1, 0, 1, 0), 'liner': numfi(0, 0, 1, 0), 'hyperbolic': numfi(-1, 1, 1, 0)}

        k1 = 1 / np.cumprod(np.power(1 + np.power(0.25, iter_index), 0.5))[-1]
        k2 = 1 / np.cumprod(np.power(1 - np.power(0.25, iter_index), 0.5))[-1]
        k1_fixed = np.array(numfi(k1, self.s_coef, self.w_coef, self.f_coef))
        k2_fixed = np.array(numfi(k2, self.s_coef, self.w_coef, self.f_coef))
        self.k = {'circle': k1_fixed, 'liner': k1_fixed, 'hyperbolic': k2_fixed}

    def sign(self, in_np):  # avoid sign(0) = 0
        return np.sign(np.sign(np.array(in_np)) + 0.1)

    def mag(self, x0, y0):
        # self.width_add = 0
        cal_shape = np.append(np.array(self.iter_num), np.array(x0.shape))
        x = numfi(np.zeros(cal_shape), self.s_in, self.w_in + self.width_add, self.f_in)
        y = numfi(np.zeros(cal_shape), self.s_in, self.w_in + self.width_add, self.f_in)
        z = numfi(np.zeros(cal_shape), self.s_in, self.w_in + self.width_add, self.f_in)
        z0 = 0
        mode = 'circle'
        d = self.sign
        u_true = self.u[mode]
        e_true = self.e[mode]
        k_true = self.k[mode]
        x[0] = numfi(x0, self.s_in, self.w_in, self.f_in)
        y[0] = numfi(y0, self.s_in, self.w_in, self.f_in)
        z[0] = numfi(z0, self.s_in, self.w_in, self.f_in)
        for i, index in enumerate(self.iter_index):
            di = numfi(-(d(x[i]) * d(y[i])), 1, 2, 0)  # -d(xi*yi)

            yi = y[i] >> index
            xi = x[i] >> index

            x[i + 1] = x[i] - u_true * di * yi
            y[i + 1] = y[i] + di * xi

        return np.array(numfi(x[-1] * (k_true), self.s_in, self.w_in + 1, self.f_in))

    def ln(self, x_in):
        self.width_add = 0
        cal_shape = np.append(np.array(self.iter_num), np.array(x_in.shape))
        x = numfi(np.zeros(cal_shape), self.s_in, self.w_in + self.width_add, self.f_in)
        y = numfi(np.zeros(cal_shape), self.s_in, self.w_in + self.width_add, self.f_in)
        z = numfi(np.zeros(cal_shape), self.s_in, self.w_in + self.width_add, self.f_in)
        x0 = x_in + 1
        y0 = x_in - 1
        z0 = 0
        mode = 'hyperbolic'
        d = self.sign
        u_true = self.u[mode]
        e_true = self.e[mode]
        x[0] = numfi(x0, self.s_in, self.w_in, self.f_in)
        y[0] = numfi(y0, self.s_in, self.w_in, self.f_in)
        z[0] = numfi(z0, self.s_in, self.w_in, self.f_in)
        for i, index in enumerate(self.iter_index):
            di = numfi(-d(y[i]), 1, 2, 0)
            if index <= 0:
                yi = y[i] - (y[i] >> (2 - index))
                xi = x[i] - (x[i] >> (2 - index))
                # print(yi)
            else:
                yi = y[i] >> index
                xi = x[i] >> index

            z[i + 1] = z[i] - di * e_true[i]
            x[i + 1] = x[i] - u_true * di * yi
            y[i + 1] = y[i] + di * xi
        # print([x])
        return np.array(numfi(z[-1] << 1, self.s_in, self.w_in, self.f_in))

def prepare_mel_rom(mel):
    # mel = mel_coeff_array
    rom_true = []
    for i in range(mel.shape[0]):
        index = np.where(mel[i] != 0)[0].tolist()
        true = mel[i][index]
        bound = np.zeros(len(index)).astype(np.int32).tolist()
        bound[-1] = 1
        rom_true.extend([x for x in zip(bound, index, true)])
    return rom_true

def mel_dot(fft_result, rom_true, mel_num):

    mel_addr = 0
    rom_cnt = 0
    mel_out = np.zeros([mel_num, fft_result.shape[1]])
    for clip in range(fft_result.shape[1]):
        rom_cnt = 0
        mel_addr = 0
        for rom_cnt in range(len(rom_true)):
            fft_addr = rom_true[rom_cnt][1]
            mel_out[mel_addr, clip] += rom_true[rom_cnt][2] * fft_result[fft_addr, clip]
            if (rom_true[rom_cnt][0]):
                mel_addr += 1
            rom_cnt += 1
    return mel_out

class qFeature():
    def __init__(self, qlist=None, feature_cfg=None):
        self.N = int(feature_cfg['n_fft'])
        self.win_out_qfmt = qlist['fft']
        self.win_qfmt = qlist['win']
        self.pow_qfmt = qlist['pow']
        self.mel_qfmt = qlist['mel']
        self.spec_qfmt = qlist['spec']
        self.logout_qfmt = qlist['log']
        self.logtan_qfmt = qlist['logtan']

        self.fft_stage = np.log2(self.N).astype(np.int32)
        self.fft_qfmt = [qlist['fft'] for x in range(self.fft_stage + 1)]

        self.my_q_fft = qFFT(self.N, self.fft_qfmt)
        #self.fft_handle = self.my_q_fft.cal_fft_with_stage
        self.fft_handle = self.my_q_fft.cal_fft
        self.sr = feature_cfg['sr']
        self.n_mel = feature_cfg['n_mels']
        self.mel_fmin = feature_cfg['fmin']
        self.mel_fmax = feature_cfg['fmax']
        self.win_length = feature_cfg['win_length']
        self.hop_length = feature_cfg['hop_length']
        self.qfft_window = get_qfft_window(n_fft=self.N, window="hann", win_length=self.win_length, qlist=self.win_qfmt)
        self.mel_filter_q = quantized_mel_filter(sr=self.sr, n_fft=self.N, n_mels=self.n_mel, fmin=self.mel_fmin,
                                                 fmax=self.mel_fmax, qlist=self.mel_qfmt)
        self.mel_rom = prepare_mel_rom(self.mel_filter_q)
        self.log_amin = 2 ** (-1 * self.spec_qfmt[2])
        self.log_points = 24
        self.xlist, self.y_qlist, self.tan_qlist = logfx_config(amin=self.log_amin, points=self.log_points,
                                                                ofmt=self.logout_qfmt, tanfmt=self.logtan_qfmt)

        # self.cordic_ln = FixedCordic( qinput = (1,25,19), qcoeff = (0,8,6), iter_index = [-5,-4,-3,-2,-1,0,1,2,3,4,4,5,6,7,8,9]) # mag out width should + 1
        self.cordic_ln = FixedCordic(qinput=(1, 25, 19), qcoeff=(0, 9, 7),
                                     iter_index=[-4, -3, -2, -1, 0, 1, 2, 3])  # mag out width should + 1
        self.cordic_mag = FixedCordic(qinput=(1, 25, 19), qcoeff=(0, 9, 7),
                                      iter_index=np.arange(16))  # mag out width should + 1

    def fbank_window(self, y):
        dtypeComplex = util.dtype_r2c(y.dtype)
        frames = frame_window(y, n_fft=self.N, hop_length=self.hop_length, center=False, pad_mode="reflect", window=self.qfft_window)
        frames = np.array(numfi(frames, self.win_out_qfmt[0], self.win_out_qfmt[1], self.win_out_qfmt[2]))
        return frames, dtypeComplex

    def fbank_fft(self, frames, dtypeComplex):
        stft_matrix = my_stft_nowindow(frames, n_fft=self.N, dtype=dtypeComplex,fft_handle=self.fft_handle)
        return stft_matrix

    def fbank_power(self, stft_matrix):
        power_fft = abs_add(stft_matrix)
        power_fft_q = np.array(numfi(power_fft, self.pow_qfmt[0], self.pow_qfmt[1], self.pow_qfmt[2]))
        return power_fft_q

    def fbank_mel(self, power_fft_q):
        spec_q = mel_dot(power_fft_q, self.mel_rom, self.n_mel) / 64.0
        spec_q_q = np.array(numfi(spec_q, self.spec_qfmt[0], self.spec_qfmt[1], self.spec_qfmt[2]))
        return spec_q_q

    def fbank_quantize(self, y, db=False):
        dtypeComplex = util.dtype_r2c(y.dtype)
        frams = frame_window(y, n_fft=self.N, hop_length=self.hop_length, center=False, pad_mode="reflect", window=self.qfft_window)
        frams = np.array(numfi(frams, self.win_out_qfmt[0], self.win_out_qfmt[1], self.win_out_qfmt[2]))

        stft_matrix = my_stft_nowindow(frams, n_fft=self.N, dtype=dtypeComplex,fft_handle=self.fft_handle)
        # frams = frame(y, n_fft=self.N, hop_length=self.hop_length, center=False, pad_mode="reflect")
        # stft_matrix = my_stft(frams, n_fft=self.N, dtype=dtypeComplex, fft_window=self.qfft_window,
        #                       fft_handle=self.fft_handle)
        # power_fft = abs_power(stft_matrix)
        # power_fft_q = np.array(numfi(power_fft,self.pow_qfmt[0],self.pow_qfmt[1],self.pow_qfmt[2]))

        power_fft = abs_add(stft_matrix)
        # power_fft = self.cordic_mag.mag(np.abs(stft_matrix.real), np.abs(stft_matrix.imag))
        power_fft_q = np.array(numfi(power_fft, self.pow_qfmt[0], self.pow_qfmt[1], self.pow_qfmt[2]))
        # print(power_fft)
        # print(self.mel_filter_q)
        spec_q = mel_dot(power_fft_q, self.mel_rom, self.n_mel) / 64.0
        # spec_q = spectrum(self.mel_filter_q, power_fft_q)/64.0
        # print(spec_q)
        spec_q_q = np.array(numfi(spec_q, self.spec_qfmt[0], self.spec_qfmt[1], self.spec_qfmt[2]))
        if db:
            # fbank = librosa.power_to_db(spec_q_q, top_db=80)
            #fbank = log10fx(spec_q_q, self.xlist, self.y_qlist, self.tan_qlist, self.log_amin ,self.log_points, self.logout_qfmt)
            fbank = self.cordic_ln.ln(spec_q_q) * 8
            # fbank = np.log(spec_q_q)*8
            fbank = np.array(numfi(fbank, self.logout_qfmt[0], self.logout_qfmt[1], self.logout_qfmt[2]))

        else:
            fbank = spec_q_q

        return fbank


if __name__ == '__main__':
    N = 256
    max_bits = np.array([5, 5, 5, 5, 5, 5, 5, 5])
    round_bits = 15 - max_bits
    round_list = [[1, 16, 10]]  # input round
    round_list += [[1, 16, int(x)] for x in round_bits]
    qlist = round_list
    my_q_fft = qFFT(N, qlist)

    audio_path = "0c40e715_nohash_0.wav"
    y, sr = librosa.load(audio_path, sr=16000)
    y = np.array(numfi(y, 1, 16, 15))
    dtypeComplex = util.dtype_r2c(y.dtype)
    data = y.astype(dtypeComplex)
    print(dtypeComplex)
    print(data.shape)
    qfft_window = get_qfft_window(n_fft=N,window="hann",win_length=N,qlist=(1, 16, 15))
    frams = frame(data.copy(), n_fft=N, hop_length=N, center=False, pad_mode="reflect")
    print(frams.shape)
    stft_matrix = my_stft(frams,n_fft=N,dtype=dtypeComplex,fft_window=qfft_window,fft_handle=my_q_fft.cal_fft)
    power_fft = abs_power(stft_matrix)
    print("power max num :" + str(np.max(power_fft)))
    power_fft_q = np.array(numfi(power_fft,0,20,10))
    mel_filter_q = quantized_mel_filter(sr=16000, n_fft=256, n_mels=40, fmin=20, fmax=4000, qlist=(1, 16, 15))
    spec = spectrum(mel_filter_q, power_fft)
    spec_q = spectrum(mel_filter_q, power_fft_q)
    print("spec max num :" + str(np.max(spec)))
    print("spec min num :" + str(np.min(spec)))

    spec_q_q = np.array(numfi(spec_q,0,16,12))
    #log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    print(np.max(np.abs(librosa.power_to_db(spec, top_db=80))))
    # plt.figure()
    # plt.scatter([x for x in range(mel_filter_q.shape[1])],mel_filter_q[39,:])
    # plt.figure()
    # plt.hist((spec).flatten(),bins=256,range=(0,1))
    # # plt.figure()
    # # plt.imshow(spec)
    # # plt.figure()
    # # plt.imshow(spec_q_q)
    # # plt.figure()
    # # plt.imshow(librosa.power_to_db(spec, top_db=80))
    # # plt.figure()
    # # plt.imshow(librosa.power_to_db(spec_q_q, top_db=None))
    # # plt.figure()
    # # plt.hist((spec_q_q).flatten())
    # # plt.figure()
    # # plt.hist(librosa.power_to_db(spec_q_q, top_db=80).flatten())
    # plt.show()

    indata = np.array([2.6,1.9,0.4,0.04])
    amin = 2 ** (-10)
    points = 32
    ofmt = [1,16,9]
    tanfmt = [1,13,6]
    xlist, y_qlist, tan_qlist = logfx_config(amin=amin, points=points,
                                                            ofmt=ofmt, tanfmt=tanfmt)
    out = log10fx(indata, xlist, y_qlist, tan_qlist, amin, ofmt)
    print(10*np.log10(amin))
    print(out)
    print(10*np.log10(indata))
    # plt.plot(xlist,ylist)
    #
    # plt.show()