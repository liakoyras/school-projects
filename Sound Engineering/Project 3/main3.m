clear;
clc;

[x1, Fs] = audioread('guitar1.wav');
[x2, Fs] = audioread('vocals.wav');

mix = 1;
room = 3;

y = reverb(x1, room, mix);
soundsc(y, 16000)