clear;
clc;

[x1, Fs] = audioread('guitar1.wav');
[x2, Fs] = audioread('vocals.wav');

gain = 11;
mix = 1;
f_sine = 0.4;
delay = 0.01;
depth = 0.003;


a = reverb(x1, 2, mix);
b = chorus(a, f_sine, delay, depth, mix, Fs);
y1 = fuzz(b, gain, mix);

soundsc(y1, Fs)