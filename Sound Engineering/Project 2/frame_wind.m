function X=frame_wind(x,frame, ovrlp)   

M=length(x);
K=floor(M/(frame*ovrlp));
X=zeros(frame,K);
W=hamming(frame);
for i=1:K-floor(1./ovrlp-1)
    indx=(i-1)*floor(frame*ovrlp);
    i;
    X(:,i)=x(indx+1:indx+frame).*W;
end