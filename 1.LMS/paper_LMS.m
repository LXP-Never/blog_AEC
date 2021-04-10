clear;clc;
snr=20;     % 信噪比
order=8;    % 自适应滤波器的阶数为8
% Hn是滤波器权重
Hn =[0.8783 -0.5806 0.6537 -0.3223 0.6577 -0.0582 0.2895 -0.2710 0.1278 ...     % ...表示换行的意思
    -0.1508 0.0238 -0.1814 0.2519 -0.0396 0.0423 -0.0152 0.1664 -0.0245 ...
    0.1463 -0.0770 0.1304 -0.0148 0.0054 -0.0381 0.0374 -0.0329 0.0313 ...
    -0.0253 0.0552 -0.0369 0.0479 -0.0073 0.0305 -0.0138 0.0152 -0.0012 ...
    0.0154 -0.0092 0.0177 -0.0161 0.0070 -0.0042 0.0051 -0.0131 0.0059 ...
    -0.0041 0.0077 -0.0034 0.0074 -0.0014 0.0025 -0.0056 0.0028 -0.0005 ...
    0.0033 -0.0000 0.0022 -0.0032 0.0012 -0.0020 0.0017 -0.0022 0.0004 -0.0011 0 0];
Hn=Hn(1:order);
mu=0.02;            % mu表示步长
N=1000;             % 设置1000个音频采样点
Loop=150;           % 150次循环
EE_LMS = zeros(N,1);    % 误差初始化
for nn=1:Loop       % epoch=150
    win_LMS = zeros(1,order);   % 权重初始化w
    error_LMS=zeros(1,N)';      % 初始化误差
    % 均匀分布的语音数据输入
    r=sign(rand(N,1)-0.5);          % shape=(1000,1)的(0,1)均匀分布-0.5，sign(n)>0=1;<0=-1
	% 声学环境语音输出：输入卷积Hn得到 输出
    output=conv(r,Hn);              % r卷积Hn,output长度=length(u)+length(v)-1
    output=awgn(output,snr,'measured');     % 加一点环境噪声：将白高斯噪声添加到信号中

    for i=order:N         % i=8：1000
      input=r(i:-1:i-order+1);  % 每次迭代取8个数据进行处理
      e_LMS = output(i)-win_LMS*input;
      win_LMS = win_LMS+2*mu*e_LMS*input';
      error_LMS(i)=error_LMS(i)+e_LMS^2;
    end
    % 把总误差相加
    EE_LMS = EE_LMS+error_LMS;

end
% 对总误差求平均值
error_LMS = EE_LMS/Loop;

figure;
error1_LMS=10*log10(error_LMS(order:N));
plot(error1_LMS,'b');              % 蓝色
axis tight;                         % 使用紧凑的坐标轴
legend('LMS算法');                  % 图例
title('LMS算法误差曲线');            % 图标题
xlabel('样本');                     % x轴标签
ylabel('误差/dB');                  % y轴标签
grid on;                            % 网格线
