% LMS�ͱ䲽��LMS(VSS LMS)�㷨��Matlab����
% Code is written by: Ray Dwaipayan
% Homepage: https://sites.google.com/view/rayd/home
clear;clc;
[d, fs] = audioread('./audio/handel.wav');      % �������(73113,1)
[x, fs_echo] = audioread('./audio/handel_echo.wav');       % ��������
M=8;                                           % ϵͳ��������ͷ��
loop=100;
mu_LMS = 0.02;          % LMS�㷨��ѧϰ��
% mu = 0.0004;        % VSS�㷨��ѧϰ��
N=length(x);

EE_mean_LMS = zeros(N,1);
y_mean_LMS = zeros(N,1);    % LMS�����ֵ
EE_mean_VSS = zeros(N,1);
y_mean_VSS = zeros(N,1);    % VSS�����ֵ
for itr=1:loop
   %% ��������ͳ�ʼģ��ϵ��
   input = zeros(M,1);           % ģ�ͳ�ͷ��ʼֵ
   % LMS�㷨ģ��(Ȩ��)����������
    model_coeff_LMS = zeros(M,1);     
    y_LMS=zeros(N,1);
    e_LMS=zeros(N,1);
    % VSS�㷨ģ��(Ȩ��)����������
    model_coeff_vss = zeros(M,1);     
    y_VSS=zeros(N,1);
    e_VSS=zeros(N,1);
    %% ѧϰ�ʵ����½�
    input_var = var(x);     % ���뷽��
    % ���mu_max��mu_min֮��Ĳ��첻����LMS��VSS LMS�㷨��������߶�����ͬ��
    mu_max = 1/(input_var*M); % �Ͻ�=1/(filter_length * input variance)
    mu_min = 0.0004;        % �½�=LMS�㷨��ѧϰ�� 0.0004;
    
    %% ����VSS�㷨�ĳ�ʼ����
    mu_VSS(M)=1;    % VSS�㷨��mu��ʼֵ
    alpha  = 0.97;
    gamma = 4.8e-4;

    for i=M:N
       %% LMS Algorithm
        input=x(i:-1:i-M+1);    % (40,1)
        y_LMS(i) = model_coeff_LMS'*input;% ģ�����(40,1)'*(40,1)=1
        e_LMS(i)=d(i)-y_LMS(i);% ���
        model_coeff_LMS = model_coeff_LMS + mu_LMS * e_LMS(i) * input;% ����Ȩ��ϵ��
       %% VSS LMS Algorithm
        y_VSS(i) = model_coeff_vss'*input;% ģ�����(40,1)'*(40,1)=1
        e_VSS(i) = d(i) - y_VSS(i);% ���
        model_coeff_vss = model_coeff_vss + mu_VSS(i) * e_VSS(i) * input;% ����Ȩ��ϵ��
        mu_VSS(i+1) = alpha * mu_VSS(i) + gamma * e_VSS(i) * e_VSS(i) ;% ʹ��VSS�㷨����muֵ
       %% mu��Լ������
        if (mu_VSS(i+1)>mu_max)
            mu_VSS(i+1)=mu_max; % max
        elseif(mu_VSS(i+1)<mu_min)
            mu_VSS(i+1)= mu_min;
        else
            mu_VSS(i+1) = mu_VSS(i+1) ;
        end
        
    end
    %% ����������LMS��VSS LMS�㷨֮��洢e_squareֵ
    err_LMS(itr,:) = e_LMS.^2;
    err_VSS(itr,:) = e_VSS.^2;
    y_mean_LMS=y_mean_LMS+y_LMS;
    y_mean_VSS=y_mean_VSS+y_VSS;
    %% ��ӡ������
    clc
    disp(char(strcat('iteration no : ',{' '}, num2str(itr) )))
end
y_mean_LMS=y_mean_LMS/loop;
y_mean_VSS=y_mean_VSS/loop;
%% �Ƚ��������
figure;
plot(10*log10(mean(err_LMS)),'-b');
hold on;
plot(10*log10(mean(err_VSS)), '-r');
title('Comparison of LMS and VSS LMS Algorithms'); xlabel('iterations');ylabel('MSE(dB)');legend('LMS Algorithm','VSS LMS Algorithm')
grid on;



