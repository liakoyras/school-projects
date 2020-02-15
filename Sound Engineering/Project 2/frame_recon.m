function y=frame_recon(X, ovrlp)   

[frame K]=size(X);

Mmax=K+floor(1./ovrlp-1);

y=zeros(1,frame*Mmax*ovrlp);
for i=1:K
    indx=(i-1)*floor(frame*ovrlp);
    y(indx+1:indx+frame)=y(indx+1:indx+frame)+X(:,i)';
end