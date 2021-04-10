% LMS�ͱ䲽��LMS(VSS LMS)�㷨��Matlab����
% Code is written by: Ray Dwaipayan
% Homepage: https://sites.google.com/view/rayd/home
clc;clear all;close all;

% �˲���ϵ��
sys_desired = [86 -294 -287 -262 -120 140 438 641 613 276 -325 -1009 -1487 ...
    -1451 -680 856 2954 5206 7106 8192 8192 7106 5206 2954 856 -680 -1451 ...
    -1487 -1009 -325 276 613 641 438 140 -120 -262 -287 -294 86] * 2^(-15);

for itr=1:100
   %% ��������ͳ�ʼģ��ϵ��
    x=randn(1,60000);                                   % ����
    model_coeff = zeros(1,length(sys_desired));         % LMS�㷨ģ��(Ȩ��)
    model_coeff_vss = zeros(1,length(sys_desired));     % VSS-LMS�㷨ģ��(Ȩ��)
    model_tap = zeros(1,length(sys_desired));           % ģ�ͳ�ͷ��ʼֵ
    %% ����40�ֱ������ذ��ϵͳ���
    noise_snr = 40;
    % filter ʹ���ɷ��Ӻͷ�ĸϵ�� sys_desired �� 1 ����������ݺ��� ���������� x �����˲���
    % awgn ���ź����40dB�ĸ�˹������
    sys_opt = filter(sys_desired,1,x)+awgn(x,noise_snr)-x;
    %% ѧϰ�ʵ����½�
    % ������Ϣ�ɴ������л�ȡ����Щֵ�������������ж����
    % R. H. Kwong and E. W. Johnston, "A variable step size LMS algorithm," in IEEE Transactions on Signal Processing, vol. 40, no. 7, pp. 1633-1642, July 1992.
    
    input_var = var(x);     % ����ķ���
    % ���mu_max��mu_min֮��Ĳ��첻����LMS��VSS LMS�㷨��������߶�����ͬ��
    mu_max = 1/(input_var*length(sys_desired)); % �Ͻ�=1/(filter_length * input variance)
    mu_LMS = 0.0004;        % LMS�㷨��ѧϰ��
    mu_min = mu_LMS;        % �½�=LMS�㷨��ѧϰ��
    
    %% ����VSS-LMS�㷨�ĳ�ʼ����
    mu_VSS(1)=1;    % VSS�㷨��mu��ʼֵ
    alpha  = 0.97;
    gamma = 4.8e-4;

    for i=1:length(x)
       %% LMS Algorithm
        model_tap=[x(i) model_tap(1:end-1)];% ģ�ͳ�ͷ(tap)ֵ(����ͷֵ����һ������)
        model_out(i) = model_tap * model_coeff';% ģ�����
        e_LMS(i)=sys_opt(i)-model_out(i);% ���
        model_coeff = model_coeff + mu_LMS * e_LMS(i) * model_tap;% ����Ȩ��ϵ��
       %% VSS LMS Algorithm
        model_out_vss(i) = model_tap * model_coeff_vss';% ģ�����
        e_vss(i) = sys_opt(i) - model_out_vss(i);% ���
        model_coeff_vss = model_coeff_vss + mu_VSS(i) * e_vss(i) * model_tap;% ����Ȩ��ϵ��
        mu_VSS(i+1) = alpha * mu_VSS(i) + gamma * e_vss(i) * e_vss(i) ;% ʹ��VSS�㷨����muֵ
       %% ��������и�����mu��Լ������
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
    err_VSS(itr,:) = e_vss.^2;
    %% ��ӡ������
    clc
    disp(char(strcat('iteration no : ',{' '}, num2str(itr) )))
end

%% �Ƚ��������
figure;
plot(10*log10(mean(err_LMS)),'-b');
hold on;
plot(10*log10(mean(err_VSS)), '-r');
title('Comparison of LMS and VSS LMS Algorithms'); xlabel('iterations');ylabel('MSE(dB)');legend('LMS Algorithm','VSS LMS Algorithm')
grid on;



