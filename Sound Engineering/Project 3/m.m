function y = m(x, gp, kp, kn)

for i = 1 : length(x)
    if x(i)>kp
        y(i) = tanh(kp) - (((tanh(kp))^2-1)/gp)*(tanh(gp*x(i)-kp));
    elseif x(i)<-kn
        y(i) = -tanh(kp) - (((tanh(kp))^2-1)/gp)*(tanh(gp*x(i)+kp));        
    else
        y(i) = tanh(x(i));
    end
end

end