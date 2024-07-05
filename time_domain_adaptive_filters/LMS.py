import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt

from tools.utils import snr2noise


def lms(x, d, N=4, mu=0.02):
    """ e(n)=d(n)-\hat{w}(n)x(n)^T
    :param x: 远端参考语音
    :param d: 近端麦克风信号(近端语音+远端回声)
    :param N: 滤波器阶数
    :param mu: 步长
    :return: 估计的近端语音
    """
    nIters = min(len(x), len(d)) - N  # 迭代点数
    u = np.zeros(N)  # 远端参考语音
    w = np.zeros(N)  # 滤波器权重
    e = np.zeros(nIters)
    for n in range(nIters):
        u[1:] = u[:-1]
        u[0] = x[n]
        e[n] = d[n] - np.dot(u, w)  # 近端麦克风信号减去估计的回声，期望等于近端语音
        w = w + 2 * mu * e[n] * u
    return e


if __name__ == "__main__":
    snr = 20  # 信噪比
    order = 8  # 自适应滤波器的阶数为8
    mu = 0.02  # mu表示步长
    N = 1000  # 设置1000个音频采样点
    Loop = 150  # 150次循环

    # 初始化滤波器权重
    Hn = np.array([
        0.8783, -0.5806, 0.6537, -0.3223, 0.6577, -0.0582, 0.2895, -0.2710, 0.1278,
        -0.1508, 0.0238, -0.1814, 0.2519, -0.0396, 0.0423, -0.0152, 0.1664, -0.0245,
        0.1463, -0.0770, 0.1304, -0.0148, 0.0054, -0.0381, 0.0374, -0.0329, 0.0313,
        -0.0253, 0.0552, -0.0369, 0.0479, -0.0073, 0.0305, -0.0138, 0.0152, -0.0012,
        0.0154, -0.0092, 0.0177, -0.0161, 0.0070, -0.0042, 0.0051, -0.0131, 0.0059,
        -0.0041, 0.0077, -0.0034, 0.0074, -0.0014, 0.0025, -0.0056, 0.0028, -0.0005,
        0.0033, -0.0000, 0.0022, -0.0032, 0.0012, -0.0020, 0.0017, -0.0022, 0.0004, -0.0011, 0, 0
    ])
    Hn = Hn[:order]

    near_speech = np.sign(np.random.rand(N) - 0.5)  # 近端语音 [-1,1,1,-1 ... -1,1,1,-1]的随机信号
    # 生成近端回声信号
    echo_signal = np.convolve(near_speech, Hn, 'full')  # 声学环境仿真
    echo_signal = snr2noise(echo_signal, snr)  # 加一点环境噪声：将白高斯噪声添加到信号中

    e = lms(near_speech, echo_signal, N=4, mu=0.02)

    plt.plot(e, 'b', label="LMS")  # 蓝色
    plt.title('LMS error')  # 图标题
    plt.xlabel('sample')  # x轴标签
    plt.ylabel('error/dB')  # y轴标签
    plt.grid()  # 网格线
    plt.show()
