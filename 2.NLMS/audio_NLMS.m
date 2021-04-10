clear;clc;
order=8;    % ����Ӧ�˲����Ľ���Ϊ8
[d, fs_orl] = audioread('./audio/handel.wav');      % �����ź�(73113,1)�޽�����������������������ź�ΪԶ����Ƶ
[x, echo] = audioread('./audio/handel_echo.wav');       % Զ�˻���

fai=0.0001;
mu=0.02;             % mu��ʾ����
N=length(x);
Loop=10;           % 150��ѭ��
EE_NLMS = zeros(N,1);       % ��ʼ������ʧ
y_NLMS = zeros(N,1);        % ��ʼ��AEC��Ƶ���
for nn=1:Loop       % epoch=150
    win_NLMS = zeros(order,1);   % Ȩ�س�ʼ��w
    y = zeros(N,1);              % ���
    error_NLMS=zeros(N,1);       % ��ʼ�����

    for i=order:N         % i=8��73113
      input=x(i:-1:i-order+1);  % ÿ�ε���ȡ8�����ݽ��д���(8,1)->(9,2)
      y(i)=win_NLMS'*input;   % (8,1)'*(8*1)=1
      error_NLMS(i) = d(i)-y(i);     % (8,1)'*(8,1)=1
      k = mu/(fai + input'*input);
      win_NLMS = win_NLMS+2*k*error_NLMS(i)*input;
      error_NLMS(i)=error_NLMS(i)^2;        % ��¼ÿ������������
    end
    % ����������
    EE_NLMS = EE_NLMS+error_NLMS;
    y_NLMS=y_NLMS+y;
end

error_NLMS = EE_NLMS/Loop;  % ���������ƽ��ֵ
y_NLMS=y_NLMS/Loop;         % �������ƽ��
audiowrite("audio/done.wav", y_NLMS, fs_orl);
sound(y_NLMS)    % ��һ���������������Ч

figure;
error1_NLMS=10*log10(error_NLMS(order:N));
plot(error1_NLMS,'b.');              % ��ɫ
axis tight;                         % ʹ�ý��յ�������
legend('NLMS�㷨');                  % ͼ��
title('NLMS�㷨�������');           % ͼ����
xlabel('����');                     % x���ǩ
ylabel('���/dB');                  % y���ǩ
grid on;                            % ������