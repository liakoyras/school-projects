function y = nonlinear(x, gn, gp, mix)   

fc = 5; % Cut off frequency
fs = 16000; % Sampling rate

[b,a] = butter(4,fc/(fs/2)); % Butterworth filter of order 4

x = 20*x;
x1 = abs(x);
lpf = filter(b,a,x1);
x2 = m((-lpf + x), gp, 2, 2);
y = x2*mix;%+ x*(1-mix);

% y = m((-lpf+20*x),gp, 2, 2)*mix+(x*20*(1-mix));

end