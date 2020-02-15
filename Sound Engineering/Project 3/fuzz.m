function y = fuzz(x, gain, mix)   


Y = sign(x).*(1 - exp(-gain*x.*sign(x)));

y = Y*mix + x*(1-mix);

end

