function map=voiced_unvoiced_det(X)

%%%%%%%  INPUT: matrix X as defined in Exercise 1
%%%%%%%  (each column contains a speech frame)
%%%%%%%  The function returns the vector map, where
%%%%%%%  map(i) = 1 when the i-th frame is unvoiced
%%%%%%%  map(i) = 0 when the i-th frame is voiced

E=sum(X.^2);
zcr=sum(abs(diff(sign(X))));

map1=zeros(size(E));
map2=map1;

map1(E<0.3*max(E))=1;
map2(zcr>0.5*max(zcr))=1;

map=map1.*map2; 