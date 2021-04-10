%% 代码来自于
%  https://blog.csdn.net/YJJat1989/article/details/21614269
%%

clear;
clc;
snr=20;     % 信噪比
order=8;    % 自适应滤波器的长度为8
Hn =[0.8783 -0.5806 0.6537 -0.3223 0.6577 -0.0582 0.2895 -0.2710 0.1278 ...     % ...表示换行的意思
    -0.1508 0.0238 -0.1814 0.2519 -0.0396 0.0423 -0.0152 0.1664 -0.0245 ...
    0.1463 -0.0770 0.1304 -0.0148 0.0054 -0.0381 0.0374 -0.0329 0.0313 ...
    -0.0253 0.0552 -0.0369 0.0479 -0.0073 0.0305 -0.0138 0.0152 -0.0012 ...
    0.0154 -0.0092 0.0177 -0.0161 0.0070 -0.0042 0.0051 -0.0131 0.0059 ...
    -0.0041 0.0077 -0.0034 0.0074 -0.0014 0.0025 -0.0056 0.0028 -0.0005 ...
    0.0033 -0.0000 0.0022 -0.0032 0.0012 -0.0020 0.0017 -0.0022 0.0004 -0.0011 0 0];
Hn=Hn(1:order);
mu=0.5;
N=1000;             % 横坐标1000个采样点
Loop=150;
EE=zeros(N,1); 
EE1=zeros(N,1);
EE2=zeros(N,1);
EE3=zeros(N,1);
for nn=1:Loop
    r=sign(rand(N,1)-0.5);          % shape=(1000,1)的(0,1)均匀分布-0.5，sign(n)>0=1;<0=-1
    output=conv(r,Hn);              % r卷积Hn,output长度=length(u)+length(v)-1
    output=awgn(output,snr,'measured');     % 将白高斯噪声添加到信号中
    win=zeros(1,order);         % 四种步长测试，四个权重——1
    win1=zeros(1,order);        % 四种步长测试，四个权重——2
    win2=zeros(1,order);        % 四种步长测试，四个权重——3
    win3=zeros(1,order);        % 四种步长测试，四个权重——4
    error=zeros(1,N)';          % 四种步长测试，四个误差——1
    error1=zeros(1,N)';         % 四种步长测试，四个误差——2
    error2=zeros(1,N)';         % 四种步长测试，四个误差——3
    error3=zeros(1,N)';         % 四种步长测试，四个误差——4
    
    % N=1000，每个采样点
    for i=order:N         % 8~1000
      input=r(i:-1:i-order+1);  % 每次迭代取8个数据进行处理 (8,1)
      y(i)=win*input;           % 权重*输入数据（初始权重是0）
      e=output(i)-win*input;    % 误差1
      e1=output(i)-win1*input;  % 误差2
      e2=output(i)-win2*input;  % 误差3
      e3=output(i)-win3*input;  % 误差4
      fai=0.0001; 
      if i<200
          mu=0.32;
      else
          mu=0.15;
      end
      % 不同步长的NLMS，w(n+1) = w(n) + ?(n)e(n)x(n)=w(n) +ηe(n)x(n)/(δ+xT(n)x(n))（η是步长。δ是一个较小的常熟，一般取0.0001）
      win=win+0.3*e*input'/(fai+input'*input);      % 步长0.3
      win1=win1+0.8*e1*input'/(fai+input'*input);   % 步长0.8
      win2=win2+1.3*e2*input'/(fai+input'*input);   % 步长1.3
      win3=win3+1.8*e3*input'/(fai+input'*input);   % 步长1.8
      error(i)=error(i)+e^2;
      error1(i)=error1(i)+e1^2;
      error2(i)=error2(i)+e2^2;
      error3(i)=error3(i)+e3^2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
      y1(i)=win1*input;
      e1=output(i)-win1*input;
      win1=win1+0.2*e1*input';                   % LMS
      error1(i)=error1(i)+e1^2;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    end
    % 把总误差相加
    EE=EE+error;
    EE1=EE1+error1;
    EE2=EE2+error2;
    EE3=EE3+error3;
end
% 对总误差求平均值
error=EE/Loop;
error1=EE1/Loop;
error2=EE2/Loop;
error3=EE3/Loop;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;     %%%% 图1
error_NLMS=10*log10(error(order:N));
error1_LMS=10*log10(error1(order:N));
plot(error_NLMS,'r');    % 红色
hold on;
plot(error1_LMS,'b.');  % 蓝色
axis tight;         % 使用紧凑的坐标轴
legend('NLMS算法','LMS算法');       % 图例
title('NLMS算法和LMS算法误差曲线');  % 图标题
xlabel('样本');                     % x轴标签
ylabel('误差/dB');                  % y轴标签
grid on;                            % 网格线

figure;     %%%% 图2
plot(win,'r');      % 权重变化，红线
hold on;
plot(Hn,'b');       % Hn(1:order)，8个数据的值，蓝线
axis tight;
grid on;

figure;     %%%% 图3
subplot(2,1,1);
plot(y,'r');        % NLMS的y(i)=win*input; % 权重*输入数据（初始权重是0）
subplot(2,1,2);
plot(y1,'b');       % LMS的y(i)=win*input; % 权重*输入数据（初始权重是0）
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;     %%%% 图4
error=10*log10(error(order:N));
error1=10*log10(error1(order:N));
error2=10*log10(error2(order:N));
error3=10*log10(error3(order:N));
hold on;
plot(error,'r');
hold on;
plot(error1,'b');
hold on;
plot(error2,'y');
hold on;
plot(error3,'g');
axis tight;
legend('η = 0.3','η = 0.8','η = 1.3','η = 1.8');
title('不同步长对NLMS算法的影响');
xlabel('样本');
ylabel('误差/dB');
grid on;

