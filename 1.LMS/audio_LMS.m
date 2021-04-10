clear;clc;
order=8;    % ����Ӧ�˲����Ľ���Ϊ8
[d, fs_orl] = audioread('./audio/handel.wav');      % �����ź�(73113,1)���޽�����������������������ź�ΪԶ����Ƶ
[x, fs_echo] = audioread('./audio/handel_echo.wav');       % Զ�˻���

mu=0.02;             % mu��ʾ���� 0.02,
N=length(x);
Loop=10;             % 150��ѭ��

EE_LMS = zeros(N,1);
y_LMS = zeros(N,1);
for nn=1:Loop       % epoch=150
    win_LMS = zeros(order,1);   % ����Ӧ�˲���Ȩ�س�ʼ��w
    y = zeros(N,1);             % ���
    error_LMS=zeros(N,1);       % ����ʼ��

    for i=order:N         % i=8��73113
      input=x(i:-1:i-order+1);  % ÿ�ε���ȡ8�����ݽ��д���(8,1)->(9,2)
      y(i)=win_LMS'*input;   % (8,1)'*(8*1)=1
      error_LMS(i) = d(i)-y(i);     % (8,1)'*(8,1)=1
      win_LMS = win_LMS+2*mu*error_LMS(i)*input;
      error_LMS(i)=error_LMS(i)^2;        % ��¼ÿ������������
    end
    % ����������
    EE_LMS = EE_LMS+error_LMS;
    y_LMS=y_LMS+y;
end
error_LMS = EE_LMS/Loop;    % ���������ƽ��ֵ
y_LMS=y_LMS/Loop;           % �������ƽ��

audiowrite("audio/done.wav", y_LMS, fs_orl);
sound(y_LMS)    % ��һ���������������Ч

figure;
error1_LMS=10*log10(error_LMS(order:N));
plot(error1_LMS,'b.');              % ��ɫ
axis tight;                         % ʹ�ý��յ�������
legend('LMS�㷨');                  % ͼ��
title('LMS�㷨�������');           % ͼ����
xlabel('����');                     % x���ǩ
ylabel('���/dB');                  % y���ǩ
grid on;                            % ������
