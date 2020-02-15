clear;
clc;

[x, Fs] = audioread('voice.wav');
%soundsc(x, Fs)

frame = 256;
ovrlp = 0.5;

X = frame_wind(x, frame, ovrlp);
[frame, nframes] = size(X);


for p = 10:35
    
    A = zeros(p+1,nframes);
    for i = 1:nframes
        A(:,i) = lpc(X(:,i),p);
        R(i,:) = xcorr(X(:,i));
        G(i) = sqrt(R(i,frame:frame+p)*A(:,i));
        e(:,i) = filter(A(:,i), 1, X(:,i));
    end
    
    s(p) = 0;
    for i = 1:nframes-1
        y = e(:,i).^2;
        s(p) = s(p) + sum(y);
    end
end

plot([10:35],s(10:35))
