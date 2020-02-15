clear;
clc;

[x, Fs] = audioread('voice.wav');
%soundsc(x, Fs)

frame = 256;
ovrlp = 0.5;

X = frame_wind(x, frame, ovrlp);
[frame, nframes] = size(X);

p = 21;
for i = 1:nframes
    A(:,i) = lpc(X(:,i),p);
    R(i,:) = xcorr(X(:,i));
    G(i) = sqrt(R(i,frame:frame+p)*A(:,i));
    E(:,i) = filter(A(:,i), 1, X(:,i));
end

e = frame_recon(E, ovrlp);
soundsc(e, Fs)

for i = 1:nframes
    Y(:,i) = filter(G(i), A(:,i), E(:,i));
end

y = frame_recon(Y, ovrlp);
soundsc(y, Fs)
