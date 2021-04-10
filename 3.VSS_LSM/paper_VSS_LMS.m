% LMS和变步长LMS(VSS LMS)算法的Matlab代码
% Code is written by: Ray Dwaipayan
% Homepage: https://sites.google.com/view/rayd/home
clc;clear all;close all;

% 滤波器系数
sys_desired = [86 -294 -287 -262 -120 140 438 641 613 276 -325 -1009 -1487 ...
    -1451 -680 856 2954 5206 7106 8192 8192 7106 5206 2954 856 -680 -1451 ...
    -1487 -1009 -325 276 613 641 438 140 -120 -262 -287 -294 86] * 2^(-15);

for itr=1:100
   %% 定义输入和初始模型系数
    x=randn(1,60000);                                   % 输入
    model_coeff = zeros(1,length(sys_desired));         % LMS算法模型(权重)
    model_coeff_vss = zeros(1,length(sys_desired));     % VSS-LMS算法模型(权重)
    model_tap = zeros(1,length(sys_desired));           % 模型抽头初始值
    %% 增加40分贝噪声地板的系统输出
    noise_snr = 40;
    % filter 使用由分子和分母系数 sys_desired 和 1 定义的有理传递函数 对输入数据 x 进行滤波。
    % awgn 给信号添加40dB的高斯白噪声
    sys_opt = filter(sys_desired,1,x)+awgn(x,noise_snr)-x;
    %% 学习率的上下界
    % 以上信息可从论文中获取，这些值是在下述论文中定义的
    % R. H. Kwong and E. W. Johnston, "A variable step size LMS algorithm," in IEEE Transactions on Signal Processing, vol. 40, no. 7, pp. 1633-1642, July 1992.
    
    input_var = var(x);     % 输入的方差
    % 如果mu_max和mu_min之间的差异不够大，LMS和VSS LMS算法的误差曲线都是相同的
    mu_max = 1/(input_var*length(sys_desired)); % 上界=1/(filter_length * input variance)
    mu_LMS = 0.0004;        % LMS算法的学习率
    mu_min = mu_LMS;        % 下界=LMS算法的学习率
    
    %% 定义VSS-LMS算法的初始参数
    mu_VSS(1)=1;    % VSS算法的mu初始值
    alpha  = 0.97;
    gamma = 4.8e-4;

    for i=1:length(x)
       %% LMS Algorithm
        model_tap=[x(i) model_tap(1:end-1)];% 模型抽头(tap)值(将抽头值右移一个样本)
        model_out(i) = model_tap * model_coeff';% 模型输出
        e_LMS(i)=sys_opt(i)-model_out(i);% 误差
        model_coeff = model_coeff + mu_LMS * e_LMS(i) * model_tap;% 更新权重系数
       %% VSS LMS Algorithm
        model_out_vss(i) = model_tap * model_coeff_vss';% 模型输出
        e_vss(i) = sys_opt(i) - model_out_vss(i);% 误差
        model_coeff_vss = model_coeff_vss + mu_VSS(i) * e_vss(i) * model_tap;% 更新权重系数
        mu_VSS(i+1) = alpha * mu_VSS(i) + gamma * e_vss(i) * e_vss(i) ;% 使用VSS算法更新mu值
       %% 检查论文中给出的mu的约束条件
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
    err_VSS(itr,:) = e_vss.^2;
    %% 打印迭代号
    clc
    disp(char(strcat('iteration no : ',{' '}, num2str(itr) )))
end

%% 比较误差曲线
figure;
plot(10*log10(mean(err_LMS)),'-b');
hold on;
plot(10*log10(mean(err_VSS)), '-r');
title('Comparison of LMS and VSS LMS Algorithms'); xlabel('iterations');ylabel('MSE(dB)');legend('LMS Algorithm','VSS LMS Algorithm')
grid on;



