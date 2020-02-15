clear;
clc;

[x, Fs] = audioread('vocals_deess.wav');
x = (x');
ovrlp = 0.5;
frame = 1024;

X = frame_wind(x, frame,ovrlp);
[frame, nframes] = size(X);

E = sum(abs(X));
ZCR = zeros(1,nframes);
for i = 1 : nframes
    for j = 2 : frame
        ZCR(1,i) = ZCR(1,i) + abs(sign(X(j-1, i)) - sign(X(j, i)));
    end
end

a = 1;
b = 0.4;
for i = 1 : nframes
    if (E(1,i) <= a*max(E)) && (ZCR(1,i) > b*max(ZCR))
        U(:, i) = X(:,i);
        V(:, i) = zeros(frame, 1);
    else
        V(:, i) = X(:,i);
        U(:, i) = zeros(frame, 1);
    end
end

U = 0.35 * U;

y1 = frame_recon(V, ovrlp);
y2 = frame_recon(U, ovrlp);
y = y1 + y2;
soundsc(y, 44100);

%checker
plot(x)
figure
hold on;
plot(y1)
plot(y2, 'r')