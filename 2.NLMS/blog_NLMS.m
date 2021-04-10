%% ����������
%  https://blog.csdn.net/YJJat1989/article/details/21614269
%%

clear;
clc;
snr=20;     % �����
order=8;    % ����Ӧ�˲����ĳ���Ϊ8
Hn =[0.8783 -0.5806 0.6537 -0.3223 0.6577 -0.0582 0.2895 -0.2710 0.1278 ...     % ...��ʾ���е���˼
    -0.1508 0.0238 -0.1814 0.2519 -0.0396 0.0423 -0.0152 0.1664 -0.0245 ...
    0.1463 -0.0770 0.1304 -0.0148 0.0054 -0.0381 0.0374 -0.0329 0.0313 ...
    -0.0253 0.0552 -0.0369 0.0479 -0.0073 0.0305 -0.0138 0.0152 -0.0012 ...
    0.0154 -0.0092 0.0177 -0.0161 0.0070 -0.0042 0.0051 -0.0131 0.0059 ...
    -0.0041 0.0077 -0.0034 0.0074 -0.0014 0.0025 -0.0056 0.0028 -0.0005 ...
    0.0033 -0.0000 0.0022 -0.0032 0.0012 -0.0020 0.0017 -0.0022 0.0004 -0.0011 0 0];
Hn=Hn(1:order);
mu=0.5;
N=1000;             % ������1000��������
Loop=150;
EE=zeros(N,1); 
EE1=zeros(N,1);
EE2=zeros(N,1);
EE3=zeros(N,1);
for nn=1:Loop
    r=sign(rand(N,1)-0.5);          % shape=(1000,1)��(0,1)���ȷֲ�-0.5��sign(n)>0=1;<0=-1
    output=conv(r,Hn);              % r���Hn,output����=length(u)+length(v)-1
    output=awgn(output,snr,'measured');     % ���׸�˹������ӵ��ź���
    win=zeros(1,order);         % ���ֲ������ԣ��ĸ�Ȩ�ء���1
    win1=zeros(1,order);        % ���ֲ������ԣ��ĸ�Ȩ�ء���2
    win2=zeros(1,order);        % ���ֲ������ԣ��ĸ�Ȩ�ء���3
    win3=zeros(1,order);        % ���ֲ������ԣ��ĸ�Ȩ�ء���4
    error=zeros(1,N)';          % ���ֲ������ԣ��ĸ�����1
    error1=zeros(1,N)';         % ���ֲ������ԣ��ĸ�����2
    error2=zeros(1,N)';         % ���ֲ������ԣ��ĸ�����3
    error3=zeros(1,N)';         % ���ֲ������ԣ��ĸ�����4
    
    % N=1000��ÿ��������
    for i=order:N         % 8~1000
      input=r(i:-1:i-order+1);  % ÿ�ε���ȡ8�����ݽ��д��� (8,1)
      y(i)=win*input;           % Ȩ��*�������ݣ���ʼȨ����0��
      e=output(i)-win*input;    % ���1
      e1=output(i)-win1*input;  % ���2
      e2=output(i)-win2*input;  % ���3
      e3=output(i)-win3*input;  % ���4
      fai=0.0001; 
      if i<200
          mu=0.32;
      else
          mu=0.15;
      end
      % ��ͬ������NLMS��w(n+1) = w(n) + ?(n)e(n)x(n)=w(n) +��e(n)x(n)/(��+xT(n)x(n))�����ǲ���������һ����С�ĳ��죬һ��ȡ0.0001��
      win=win+0.3*e*input'/(fai+input'*input);      % ����0.3
      win1=win1+0.8*e1*input'/(fai+input'*input);   % ����0.8
      win2=win2+1.3*e2*input'/(fai+input'*input);   % ����1.3
      win3=win3+1.8*e3*input'/(fai+input'*input);   % ����1.8
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
    % ����������
    EE=EE+error;
    EE1=EE1+error1;
    EE2=EE2+error2;
    EE3=EE3+error3;
end
% ���������ƽ��ֵ
error=EE/Loop;
error1=EE1/Loop;
error2=EE2/Loop;
error3=EE3/Loop;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;     %%%% ͼ1
error_NLMS=10*log10(error(order:N));
error1_LMS=10*log10(error1(order:N));
plot(error_NLMS,'r');    % ��ɫ
hold on;
plot(error1_LMS,'b.');  % ��ɫ
axis tight;         % ʹ�ý��յ�������
legend('NLMS�㷨','LMS�㷨');       % ͼ��
title('NLMS�㷨��LMS�㷨�������');  % ͼ����
xlabel('����');                     % x���ǩ
ylabel('���/dB');                  % y���ǩ
grid on;                            % ������

figure;     %%%% ͼ2
plot(win,'r');      % Ȩ�ر仯������
hold on;
plot(Hn,'b');       % Hn(1:order)��8�����ݵ�ֵ������
axis tight;
grid on;

figure;     %%%% ͼ3
subplot(2,1,1);
plot(y,'r');        % NLMS��y(i)=win*input; % Ȩ��*�������ݣ���ʼȨ����0��
subplot(2,1,2);
plot(y1,'b');       % LMS��y(i)=win*input; % Ȩ��*�������ݣ���ʼȨ����0��
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;     %%%% ͼ4
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
legend('�� = 0.3','�� = 0.8','�� = 1.3','�� = 1.8');
title('��ͬ������NLMS�㷨��Ӱ��');
xlabel('����');
ylabel('���/dB');
grid on;

