function yfinal=chorus(x,f_sine,delay,depth,mix,fs)

BL=0.7;
FB=-0.7;
FF=1;

N=length(x);
y=zeros(1,N);
yfinal=zeros(1,N);

ndelay=floor(delay*fs);
ndepth=floor(depth*fs);

M=zeros(1,N);
for i=1:N
    M(i)=floor(ndelay+ndepth*(0.5+0.5*sin(2*pi*f_sine*(i/fs))));
end
Mmax=max(M);

xh=zeros(1,N);
for i=Mmax+1:N
    xh(i)=x(i)+FB*xh(i-M(i));
   
end
for i=Mmax+1:N
     y(i)=BL*xh(i)+FF*xh(i-M(i));
end
   
for i=1:N
     yfinal(i)=mix*y(i)+(1-mix)*x(i);
end
    