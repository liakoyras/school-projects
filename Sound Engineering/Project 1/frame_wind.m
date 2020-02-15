function X = frame_wind(x, frame, ovrlp)
    if nargin == 2 || ovrlp < 0 || ovrlp > 1
      ovrlp = 0.5;
    end

    W = hamming(frame);
    N = length(x);
    nframes = N/(ovrlp*frame);

    if (mod(N, frame) ~= 0)
        z = frame - mod(N, frame);
        for k = N+1 : N+z
            x(1,k) = 0;
        end
    else
        z = frame/2;
        for k = N+1 : N+z
            x(1,k) = 0;
        end
    end    

    for i = 1: nframes
        for j = 1: frame
            index = (i-1)*ovrlp*frame + j;
            X(j,i) = W(j)*x(index);
        end
    end
end
