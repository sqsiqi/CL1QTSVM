function  [V1,V2,VV1,VV2,f,FF1,FF2,kk]=CL1QTSVM(c,C,X,Y)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CL1QTSVM: A kernel-free quadratic surface twin support vector machine with capped $L_1$-norm distance metric for robust classification
% Use method: Import the data first, and then run the "meanCL1QTSVMtest" procedure
% Input:
%    X: Training data.
%    Y: Training data labels. (Y must include 1 and -1)
% Parameters:
% C: The loss term regularization parameter
% c: The structural risk term regularization parameter
% Output:
% V1: The positive decision function parameter omega_+
% FF1 and FF2 are used to analyze the convergence of the two objective function
% Reference:
%    Qi Si, Zhi-Xia Yang, et. al. "A kernel-free quadratic surface twin support vector machine with 
%      capped $L_1$-norm distance metric for robust classification" Submitted 2025
%    Written by Qi Si (sq1418426889@163.com) 
% The code certainly requires a lot of refinement, so if you have any good suggestions, please contact Qi Si.  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
A=[X,Y];
x1=A(A(:,end)==1,1:end-1);
x2=A(A(:,end)==-1,1:end-1);
[g1]=X_pro(x1);
[g2]=X_pro(x2);
ML=size(g1,2);

m1=size(g1,1);
m2=size(g2,1);
e1=ones(m1,1);
e2=ones(m2,1);

N=10;
t=1;
% Z10=zeros(size(g1,2),1);
% Z20=zeros(size(g2,2),1);
Z10=1e-3*ones(size(g1,2),1);
Z20=1e-3*ones(size(g2,2),1);
% Z10=-ones(size(g1,2),1);
% Z20=-ones(size(g2,2),1);
F_old=zeros(m1,m1);
G_old=zeros(m2,m2);
M_old=zeros(m2,m2);
Z_old=zeros(m1,m1);
FF1=zeros(N,1);
FF2=zeros(N,1);
VV1=zeros(size(g1,2),N);
VV2=zeros(size(g2,2),N);
%Solve v_1
while(t<=N)
    for i=1:size(g1,1)
        if abs(g1(i,:)*Z10)<=1e-5
            F_old(i,i)=1/abs(g1(i,:)*Z10);
        else
            F_old(i,i)=1e-5;
        end
    end
    for j=1:m2
        if abs(1+g2(j,:)*Z10)<=1e-5
            G_old(j,j)=1/abs(1+g2(j,:)*Z10);
        else
            G_old(j,j)=1e-5;
        end
    end
    MM1=zeros(m1,1);
    for i=1:m1
        MM1(i,1)=min(abs(g1(i,:)*Z10),1e-5);
    end
    MMM1=sum(MM1,1);
    MM2=zeros(m2,1);
    for j=1:m2
        MM2(j,1)=min(abs(1+g2(j,:)*Z10),1e-5);
    end
    MMM2=sum(MM2,1);
    %w SMW
    if ML>m2
        Y=1/c*(eye(ML)-g1'*((c*inv(F_old)+g1*g1')\g1));
        EE=C*(Y-Y*g2'*((1/C*inv(G_old)+g2*Y*g2')\g2)*Y);
        Z1=-EE*g2'*G_old*e2;
    else
        Z1=-C*(c*eye(size(g1,2))+C*g2'*G_old*g2+g1'*F_old*g1)\g2'*G_old*e2;
    end
  
    %w/o SMW
    %Z1=-C*(c*eye(size(g1,2))+C*g2'*G_old*g2+g1'*F_old*g1)\g2'*G_old*e2;
    Z10=Z1;
    ff1=1/2*MMM1+1/2*c*(Z10'*Z10)+1/2*C*MMM2;
    FF1(t,1)=ff1;
    VV1(:,t)=Z10;
    t=t+1;
end
V1=Z1;
t=1;
%Solve v_2
while(t<=N)
    for i=1:size(g2,1)
        if abs(g2(i,:)*Z20)<=1e-5
            M_old(i,i)=1/abs(g2(i,:)*Z20);
        else
            M_old(i,i)=1e-5;
        end
    end 
    for j=1:m1
        if abs(1-g1(j,:)*Z20)<=1e-5
            Z_old(j,j)=1/abs(1-g1(j,:)*Z20);
        else
            Z_old(j,j)=1e-5;
        end
    end
    MM1=zeros(m1,1);
    for i=1:m1
        MM1(i,1)=min(abs(g1(i,:)*Z10),1e-5);
    end
    MMM1=sum(MM1,1);
    MM2=zeros(m2,1);
    for j=1:m2
        MM2(j,1)=min(abs(1+g2(j,:)*Z10),1e-5);
    end
    MMM2=sum(MM2,1);
    %w SMW
    if ML>m1
        Y=1/c*(eye(ML)-g2'*((c*inv(M_old)+g2*g2')\g2));
        EE=C*(Y-Y*g1'*((1/C*inv(Z_old)+g1*Y*g1')\g1)*Y);
        Z2=EE*g1'*Z_old*e1;
    else
        Z2=C*(c*eye(size(g1,2))+C*g1'*Z_old*g1+g2'*M_old*g2)\g1'*Z_old*e1;
    end
    
    %w/o SMW 
    %Z2=C*(c*eye(size(g1,2))+C*g1'*Z_old*g1+g2'*M_old*g2)\g1'*Z_old*e1;
    Z20=Z2;
    ff2=1/2*MMM1+1/2*c*(Z20'*Z20)+1/2*C*MMM2;
    FF2(t,1)=ff2;
    VV2(:,t)=Z20;
    t=t+1;
end
V2=Z2;
kk=N;
f=FF1+FF2;
end