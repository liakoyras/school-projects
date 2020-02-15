clear;
clc;

[x1, Fs] = audioread('voice1.wav');
%soundsc(x1, 8000)
[x2, Fs] = audioread('voice2.wav');
%soundsc(x2, 8000)
[x, Fs] = audioread('voice2align.wav');
%soundsc(x, 8000)

% plot(x1)
% figure
% plot(x2)
% figure
% plot(x)


frame = 256;
ovrlp = 0.5;

X1 = frame_wind(x1, frame, ovrlp);
[frame, nframes] = size(X1);
X2 = frame_wind(x, frame, ovrlp);
[frame, nframes] = size(X2);

p = 21;
for i = 1:nframes
    A1(:,i) = lpc(X1(:,i),p);
    R1(i,:) = xcorr(X1(:,i));
    G1(i) = sqrt(R1(i,frame:frame+p)*A1(:,i));
    E1(:,i) = filter(A1(:,i), 1, X1(:,i));
end

for i = 1:nframes
    A2(:,i) = lpc(X2(:,i),p);
    R2(i,:) = xcorr(X2(:,i));
    G2(i) = sqrt(R2(i,frame:frame+p)*A2(:,i));
    E2(:,i) = filter(A2(:,i), 1, X2(:,i));
end


for i = 1:nframes
    Y1(:,i) = filter(G2(i), A2(:,i), E1(:,i));
    Y2(:,i) = filter(G1(i), A1(:,i), E2(:,i));
end

y1 = frame_recon(Y1, ovrlp);
y2 = frame_recon(Y2, ovrlp);

soundsc(y1, Fs)
soundsc(y2, Fs)