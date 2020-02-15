clear;
clc;

[x, Fs] = audioread('guit2.wav');
x = (x');
ovrlp = 0.5;
frame = 256;
X = frame_wind(x, frame,ovrlp);
[frame, nframes] = size(X);

E = sum(abs(X));
ZCR = zeros(1,nframes);
for i = 1 : nframes
    for j = 2 : frame
        ZCR(1,i) = ZCR(1,i) + abs(sign(X(j-1, i)) - sign(X(j, i)));
    end
end

a = 0.4;
b = 0.4;
for i = 1 : nframes
    if (E(1,i) < a*max(E)) && (ZCR(1,i) > b*max(ZCR))
        U(:, i) = X(:,i);
        V(:, i) = zeros(frame, 1);
    else
        V(:, i) = X(:,i);
        U(:, i) = zeros(frame, 1); 
    end
end

y1 = frame_recon(V, ovrlp);
soundsc(y1, 16000)
y2 = frame_recon(U, ovrlp);
soundsc(y2, 16000)
plot(y1)
figure
plot(y2)
y = y1 + y2;
soundsc(y, 16000)
figure
plot(y)
