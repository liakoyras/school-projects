function y = reverb(x, room, mix)   

load('rooms.mat')

if room == 1
    w = conv(x, h1);
    x(numel(w)) = 0;
elseif room == 2
    w = conv(x, h2);
    x(numel(w)) = 0;
elseif room == 3
    w = conv(x, h3);
    x(numel(w)) = 0;
end

y = w*mix + x*(1-mix);

end