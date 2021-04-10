clc;clear;

snr=20;     % �����
order=8;    % ����Ӧ�˲����Ľ���Ϊ8
% ������ѧ����������Ӧ
Hn =[0.8783 -0.5806 0.6537 -0.3223 0.6577 -0.0582 0.2895 -0.2710 0.1278 ...     % ...��ʾ���е���˼
    -0.1508 0.0238 -0.1814 0.2519 -0.0396 0.0423 -0.0152 0.1664 -0.0245 ...
    0.1463 -0.0770 0.1304 -0.0148 0.0054 -0.0381 0.0374 -0.0329 0.0313 ...
    -0.0253 0.0552 -0.0369 0.0479 -0.0073 0.0305 -0.0138 0.0152 -0.0012 ...
    0.0154 -0.0092 0.0177 -0.0161 0.0070 -0.0042 0.0051 -0.0131 0.0059 ...
    -0.0041 0.0077 -0.0034 0.0074 -0.0014 0.0025 -0.0056 0.0028 -0.0005 ...
    0.0033 -0.0000 0.0022 -0.0032 0.0012 -0.0020 0.0017 -0.0022 0.0004 -0.0011 0 0];
Hn=Hn(1:order);
mu=0.5;             % mu��ʾ����
N=1000;             % 1000����Ƶ������
Loop=150;           % 150��ѭ��
EE_NLMS=zeros(N,1); % ��ʼ�������
for nn=1:Loop       % epoch=150
    win_NLMS=zeros(1,order);         % Ȩ�س�ʼ��w
    error_NLMS=zeros(1,N)';     % ��ʼ�����
    % ���ȷֲ�������ֵ
    r=sign(rand(N,1)-0.5);          % shape=(1000,1)��(0,1)���ȷֲ�-0.5��sign(n)>0=1;<0=-1
	% ��ѧ������Ƶ�����������Hn�õ� ���
    output=conv(r,Hn);              % r���Hn,output����=length(u)+length(v)-1
    output=awgn(output,snr,'measured');     % ���׸�˹������ӵ��ź���

    % N=1000��ÿ��������
    for i=order:N         % i=8��1000
      input=r(i:-1:i-order+1);  % ÿ�ε���ȡ8�����ݽ��д���
      e_NLMS = output(i)-win_NLMS*input;
      win_NLMS=win_NLMS+e_NLMS*input'/(input'*input);   % NLMS����Ȩ��
      error_NLMS(i)=error_NLMS(i)+e_NLMS^2;
    end
    
    EE_NLMS=EE_NLMS+error_NLMS;     % ����������
end
% ���������ƽ��ֵ
error_NLMS=EE_NLMS/Loop;

figure;
error_NLMS=10*log10(error_NLMS(order:N));
plot(error_NLMS,'r');       % ��ɫ
axis tight;                 % ʹ�ý��յ�������
legend('NLMS�㷨');           % ͼ��
title('NLMS�㷨�������');     % ͼ����
xlabel('����');                     % x���ǩ
ylabel('���/dB');                  % y���ǩ
grid on;                            % ������


