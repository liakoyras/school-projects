function y = frame_recon(X, ovrlp)
    if nargin == 2 || ovrlp < 0 || ovrlp > 1
      ovrlp = 0.5;
    end

    [frame, nframes] = size(X);
    N = frame*(nframes + 1)*ovrlp;

    y = zeros(1,N);

    for i = 1 : nframes
        for j = 1 : frame
            index = (i-1)*ovrlp*frame + j;
            y(1,index) =  y(1,index) + X(j,i);
        end 
    end
end

