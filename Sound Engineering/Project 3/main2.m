clear;
clc;

[x1, Fs] = audioread('guitar1.wav');
[x2, Fs] = audioread('vocals.wav');
[x3, Fs] = audioread('guitar_kids.wav');

mix = 1;
f_sine = 0.4;
delay = 0.01;
depth = 0.003;

y1 = chorus(x1, f_sine, delay, depth, mix, 16000);
y3 = chorus(x3, f_sine, delay, depth, mix, 44100);
soundsc(x3, 44100)

figure
specgram(x1)
figure
specgram(y1)