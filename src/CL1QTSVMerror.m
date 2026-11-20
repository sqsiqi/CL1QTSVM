function [err,f1]=CL1QTSVMerror(Xv,Yv,v1,v2)
[~,n]=size(Xv);
% w/o qvec dvec
d=(n^2+n)/2;
ww1=v1(1:d);
ind=find(tril(ones(n)));      
W1=zeros(n,n);
W1(ind)=ww1;                    
W2=W1-diag(diag(W1));        
w1=W2+W1';
%求得b
b1=v1(d+1:d+n);
%求得c
c1=v1(d+n+1);

% with qvec dvec
% d=n;
% ww1=v1(1:d);
% w1=diag(ww1);

%w/o qvec dvec
ww2=v2(1:d);
ind=find(tril(ones(n)));     
W1=zeros(n,n);
W1(ind)=ww2;                   
W2=W1-diag(diag(W1));        
w2=W2+W1';

% %with qvec dvec
% d=n;
% ww2=v2(1:d);
% w2=diag(ww2);

%求得b
b2=v2(d+1:d+n);
%求得c
c2=v2(d+n+1);
m=size(Xv,1);
d1=zeros(m,1);
for i=1:m
    dd=abs(1/2*Xv(i,:)*w1*Xv(i,:)'+Xv(i,:)*b1+c1)/norm(w1*Xv(i,:)'+b1);
    d1(i,1)=dd;
end

d2=zeros(m,1);
for i=1:m
    dd=abs(1/2*Xv(i,:)*w2*Xv(i,:)'+Xv(i,:)*b2+c2)/norm(w2*Xv(i,:)'+b2);
    d2(i,1)=dd;
end
y=d1-d2;
y(y<0)=1;
y(y~=1)=-1;
preY=y;
err=sum(preY ~=Yv)/size(Xv,1);
f1 = calculate_f1_score(Yv,preY);
