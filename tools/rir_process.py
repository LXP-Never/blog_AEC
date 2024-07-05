# Author:凌逆战
# -*- coding:utf-8 -*-
import numpy as np
import soundfile as sf
from pyroomacoustics import room

# 设定参数
fs = 16000  # 采样率
duration = 3  # 语音持续时间（秒）

# 生成随机语音信号
wav,wav_sr = sf.read("./wav_data/TIMIT.wav")

# 设定房间参数
room_dim = [5, 4, 3]  # 房间尺寸 (m)
mic_pos = np.array([[2, 2, 1.5]]).T  # 麦克风位置 (m)
source_pos = np.array([[3, 3, 1.5]]).T  # 源信号位置 (m)

# 模拟房间声学
room_sim = room.ShoeBox(room_dim, fs=fs)
room_sim.add_microphone_array(room.MicrophoneArray(mic_pos, fs=fs))
room_sim.add_source(source_pos, signal=wav)
room_sim.simulate()

# 获取混响后的语音信号
reverberant_signal = room_sim.mic_array.signals[0]

# 保存混响后的语音信号
sf.write("./wav_data/TIMIT_echo.wav", reverberant_signal, fs)