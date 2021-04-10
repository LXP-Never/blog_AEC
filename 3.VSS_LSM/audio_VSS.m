% LMS和变步长LMS(VSS LMS)算法的Matlab代码
% Code is written by: Ray Dwaipayan
% Homepage: https://sites.google.com/view/rayd/home
clear;clc;
[d, fs] = audioread('./audio/handel.wav');      % 期望输出(73113,1)
[x, fs_echo] = audioread('./audio/handel_echo.wav');       % 回声输入
M=8;                                           % 系统阶数，抽头数
loop=100;
mu_LMS = 0.02;          % LMS算法的学习率
% mu = 0.0004;        % VSS算法的学习率
N=length(x);

EE_mean_LMS = zeros(N,1);
y_mean_LMS = zeros(N,1);    % LMS输出均值
EE_mean_VSS = zeros(N,1);
y_mean_VSS = zeros(N,1);    % VSS输出均值
for itr=1:loop
   %% 定义输入和初始模型系数
   input = zeros(M,1);           % 模型抽头初始值
   % LMS算法模型(权重)、输出、误差
    model_coeff_LMS = zeros(M,1);     
    y_LMS=zeros(N,1);
    e_LMS=zeros(N,1);
    % VSS算法模型(权重)、输出、误差
    model_coeff_vss = zeros(M,1);     
    y_VSS=zeros(N,1);
    e_VSS=zeros(N,1);
    %% 学习率的上下界
    input_var = var(x);     % 输入方差
    % 如果mu_max和mu_min之间的差异不够大，LMS和VSS LMS算法的误差曲线都是相同的
    mu_max = 1/(input_var*M); % 上界=1/(filter_length * input variance)
    mu_min = 0.0004;        % 下界=LMS算法的学习率 0.0004;
    
    %% 定义VSS算法的初始参数
    mu_VSS(M)=1;    % VSS算法的mu初始值
    alpha  = 0.97;
    gamma = 4.8e-4;

    for i=M:N
       %% LMS Algorithm
        input=x(i:-1:i-M+1);    % (40,1)
        y_LMS(i) = model_coeff_LMS'*input;% 模型输出(40,1)'*(40,1)=1
        e_LMS(i)=d(i)-y_LMS(i);% 误差
        model_coeff_LMS = model_coeff_LMS + mu_LMS * e_LMS(i) * input;% 更新权重系数
       %% VSS LMS Algorithm
        y_VSS(i) = model_coeff_vss'*input;% 模型输出(40,1)'*(40,1)=1
        e_VSS(i) = d(i) - y_VSS(i);% 误差
        model_coeff_vss = model_coeff_vss + mu_VSS(i) * e_VSS(i) * input;% 更新权重系数
        mu_VSS(i+1) = alpha * mu_VSS(i) + gamma * e_VSS(i) * e_VSS(i) ;% 使用VSS算法更新mu值
       %% mu的约束条件
        if (mu_VSS(i+1)>mu_max)
            mu_VSS(i+1)=mu_max; % max
        elseif(mu_VSS(i+1)<mu_min)
            mu_VSS(i+1)= mu_min;
        else
            mu_VSS(i+1) = mu_VSS(i+1) ;
        end
        
    end
    %% 在完整运行LMS和VSS LMS算法之后存储e_square值
    err_LMS(itr,:) = e_LMS.^2;
    err_VSS(itr,:) = e_VSS.^2;
    y_mean_LMS=y_mean_LMS+y_LMS;
    y_mean_VSS=y_mean_VSS+y_VSS;
    %% 打印迭代号
    clc
    disp(char(strcat('iteration no : ',{' '}, num2str(itr) )))
end
y_mean_LMS=y_mean_LMS/loop;
y_mean_VSS=y_mean_VSS/loop;
%% 比较误差曲线
figure;
plot(10*log10(mean(err_LMS)),'-b');
hold on;
plot(10*log10(mean(err_VSS)), '-r');
title('Comparison of LMS and VSS LMS Algorithms'); xlabel('iterations');ylabel('MSE(dB)');legend('LMS Algorithm','VSS LMS Algorithm')
grid on;



