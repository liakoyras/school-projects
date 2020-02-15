clear;
clc;

[x, Fs] = audioread('voice.wav');
%soundsc(x, 16000)

frame = 256;
ovrlp = 0.5;

X = frame_wind(x, frame, ovrlp);
[frame, nframes] = size(X);

T = 1/125;
k = floor(Fs/(1/T));

Unvoiced = voiced_unvoiced_det(X);

voiced_frame = zeros(1,frame);
for i = 1:frame
    if mod(i,k)==1
        voiced_frame(1,i) = 1;
    end
end
unvoiced_frame = 0.1 * randn(1,frame);

p = 21;
for i = 1:nframes
    A(:,i) = lpc(X(:,i),p);
    R(i,:) = xcorr(X(:,i));
    G(i) = sqrt(R(i,frame:frame+p)*A(:,i));
end

for i = 1:nframes
    if Unvoiced(i) == 1
        Y(:,i) = filter(G(i), A(:,i), unvoiced_frame);
    else
        Y(:,i) = filter(G(i), A(:,i), voiced_frame);
    end
end

y = frame_recon(Y, ovrlp);
soundsc(y, 16000)
