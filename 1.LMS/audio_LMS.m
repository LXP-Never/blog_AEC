clear;clc;
order=8;    % 自适应滤波器的阶数为8
[d, fs_orl] = audioread('./audio/handel.wav');      % 期望信号(73113,1)，无近端语音，所以这里的期望信号为远端音频
[x, fs_echo] = audioread('./audio/handel_echo.wav');       % 远端回声

mu=0.02;             % mu表示步长 0.02,
N=length(x);
Loop=10;             % 150次循环

EE_LMS = zeros(N,1);
y_LMS = zeros(N,1);
for nn=1:Loop       % epoch=150
    win_LMS = zeros(order,1);   % 自适应滤波器权重初始化w
    y = zeros(N,1);             % 输出
    error_LMS=zeros(N,1);       % 误差初始化

    for i=order:N         % i=8：73113
      input=x(i:-1:i-order+1);  % 每次迭代取8个数据进行处理，(8,1)->(9,2)
      y(i)=win_LMS'*input;   % (8,1)'*(8*1)=1
      error_LMS(i) = d(i)-y(i);     % (8,1)'*(8,1)=1
      win_LMS = win_LMS+2*mu*error_LMS(i)*input;
      error_LMS(i)=error_LMS(i)^2;        % 记录每个采样点的误差
    end
    % 把总误差相加
    EE_LMS = EE_LMS+error_LMS;
    y_LMS=y_LMS+y;
end
error_LMS = EE_LMS/Loop;    % 对总误差求平均值
y_LMS=y_LMS/Loop;           % 对输出求平均

audiowrite("audio/done.wav", y_LMS, fs_orl);
sound(y_LMS)    % 听一听回声消除后的音效

figure;
error1_LMS=10*log10(error_LMS(order:N));
plot(error1_LMS,'b.');              % 蓝色
axis tight;                         % 使用紧凑的坐标轴
legend('LMS算法');                  % 图例
title('LMS算法误差曲线');           % 图标题
xlabel('样本');                     % x轴标签
ylabel('误差/dB');                  % y轴标签
grid on;                            % 网格线
