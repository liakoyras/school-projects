clear;
clc;

% Check that the two sounds have no discernible difference
[x, Fs] = audioread('guit2.wav');
x = (x');
soundsc(x, 16000)
frame = 256;
ovrlp = 0.5;
X = frame_wind(x, frame, ovrlp);
y = frame_recon(X, ovrlp);
soundsc(y, 16000)

% Compare the spectrogram we created with the one produced by MATLAB
F = fft(X);
imagesc(20*log10(abs(F)));
F(1:frame/2,:)=[];
figure
imagesc(20*log10(abs(F)));
figure
specgram(x)