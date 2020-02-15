clear;
clc;

[x, Fs] = audioread('voice.wav');
%soundsc(x, 16000)

frame = 256;
ovrlp = 0.5;
p = 21;

X = frame_wind(x, frame, ovrlp);
[frame, nframes] = size(X);

for i = 1:nframes
    A(:,i) = lpc(X(:,i),p);
    R(i,:) = xcorr(X(:,i));
    G(i) = sqrt(R(i,frame:frame+p)*A(:,i));
end