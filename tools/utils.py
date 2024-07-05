# -*- coding:utf-8 -*-
# Author:凌逆战 | Never
# Date: 
"""

"""
import numpy as np

def snr2noise(clean, SNR):
    """
    :param clean: 纯净语音
    :param far_echo: 噪音
    :param SER: 指定的SNR
    :return: 根据指定的SNR求带噪语音(纯净语音+噪声)
    """
    # noise等于白噪
    noise = np.random.normal(size=clean.shape)  # 产生与纯净语音相同长度的白噪声
    p_clean = np.mean(clean ** 2)  # 纯净语音功率
    p_noise = np.mean(noise ** 2)  # 噪声功率

    scalar = np.sqrt(p_clean / (10 ** (SNR / 10)) / (p_noise + np.finfo(np.float32).eps))
    noisy = clean + scalar * noise

    return noisy
