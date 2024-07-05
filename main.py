# -*- coding:utf-8 -*-
# Author:凌逆战 | Never
# Date: 2024/7/5
"""

"""
import numpy as np
import librosa
import soundfile as sf
import pyroomacoustics as pra

from time_domain_adaptive_filters.LMS import lms

def main():
    x, sr = librosa.load('samples/female.wav', sr=8000)     # 远端语音 (114160,)
    d, sr = librosa.load('samples/male.wav', sr=8000)       # 近端语音 (64000,)

    rt60_tgt = 0.08
    room_dim = [2, 2, 2]

    e_absorption, max_order = pra.inverse_sabine(rt60_tgt, room_dim)
    room = pra.ShoeBox(room_dim, fs=sr, materials=pra.Material(e_absorption), max_order=max_order)
    room.add_source([1.5, 1.5, 1.5])
    room.add_microphone([0.1, 0.5, 0.1])
    room.compute_rir()
    rir = room.rir[0][0]
    rir = rir[np.argmax(rir):]

    y = np.convolve(x, rir)     # 远端语音加回声
    scale = np.sqrt(np.mean(x ** 2)) / np.sqrt(np.mean(y ** 2))
    y = y * scale   # 让y的幅度和x的幅度相同 (115029, )

    L = max(len(y), len(d))
    y = np.pad(y, [0, L - len(y)])  # (115029, )
    d = np.pad(d, [L - len(d), 0])# (115029, )
    x = np.pad(x, [0, L - len(x)])# (115029, )
    d = d + y   # 近端语音+远端回声

    sf.write('samples/x.wav', x, sr, subtype='PCM_16')
    sf.write('samples/d.wav', d, sr, subtype='PCM_16')

    print("processing time domain adaptive filters.")

    e = lms(x, d, N=256, mu=0.1)
    e = np.clip(e, -1, 1)
    sf.write('samples/lms.wav', e, sr, subtype='PCM_16')







