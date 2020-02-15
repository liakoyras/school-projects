clear;
clc;

%erotima a
[x, Fs] = audioread('guit2.wav');
x = (x');
ovrlp = 0.5;
frame = 256;
X = frame_wind(x, frame,ovrlp);

[frame, nframes] = size(X);
counter = 0;
E = sum(abs(X));
plot(E)
for i = 1 : nframes
    if (E(1,i) > 1.4)
        Y(:, i) = X(:,i);
    else
        Y(:, i) = zeros(frame, 1);
        counter = counter +1;
    end
end

y = frame_recon(Y, ovrlp);
sav = counter/nframes
soundsc(y, 16000)