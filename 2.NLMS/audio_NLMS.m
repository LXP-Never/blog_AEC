clear;clc;
order=8;    % 自适应滤波器的阶数为8
[d, fs_orl] = audioread('./audio/handel.wav');      % 期望信号(73113,1)无近端语音，所以这里的期望信号为远端音频
[x, echo] = audioread('./audio/handel_echo.wav');       % 远端回声

fai=0.0001;
mu=0.02;             % mu表示步长
N=length(x);
Loop=10;           % 150次循环
EE_NLMS = zeros(N,1);       % 初始化总损失
y_NLMS = zeros(N,1);        % 初始化AEC音频输出
for nn=1:Loop       % epoch=150
    win_NLMS = zeros(order,1);   % 权重初始化w
    y = zeros(N,1);              % 输出
    error_NLMS=zeros(N,1);       % 初始化误差

    for i=order:N         % i=8：73113
      input=x(i:-1:i-order+1);  % 每次迭代取8个数据进行处理，(8,1)->(9,2)
      y(i)=win_NLMS'*input;   % (8,1)'*(8*1)=1
      error_NLMS(i) = d(i)-y(i);     % (8,1)'*(8,1)=1
      k = mu/(fai + input'*input);
      win_NLMS = win_NLMS+2*k*error_NLMS(i)*input;
      error_NLMS(i)=error_NLMS(i)^2;        % 记录每个采样点的误差
    end
    % 把总误差相加
    EE_NLMS = EE_NLMS+error_NLMS;
    y_NLMS=y_NLMS+y;
end

error_NLMS = EE_NLMS/Loop;  % 对总误差求平均值
y_NLMS=y_NLMS/Loop;         % 对输出求平均
audiowrite("audio/done.wav", y_NLMS, fs_orl);
sound(y_NLMS)    % 听一听回声消除后的音效

figure;
error1_NLMS=10*log10(error_NLMS(order:N));
plot(error1_NLMS,'b.');              % 蓝色
axis tight;                         % 使用紧凑的坐标轴
legend('NLMS算法');                  % 图例
title('NLMS算法误差曲线');           % 图标题
xlabel('样本');                     % x轴标签
ylabel('误差/dB');                  % y轴标签
grid on;                            % 网格线