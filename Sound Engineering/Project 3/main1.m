clear;
clc;

[x1, Fs] = audioread('guitar1.wav');
[x2, Fs] = audioread('vocals.wav');

mix = 1;
gain = 25;
gn = 15;
gp = 25;

y1 = fuzz(x1, mix, gain);
y2 = fuzz(x2, mix, gain);

% figure
% specgram(x1)
% figure
% specgram(y1)

yn1 = nonlinear(x1, gn, gp, mix);
yn2 = nonlinear(x2, gn, gp, mix);

figure
specgram(yn2)

%soundsc(yn2, 16000)