function [g]=X_pro(X_train)
%w/o qvec and dvec
[m,n]=size(X_train);      
g=zeros(m,(n.^2+3.*n+2)./2);
P=zeros(n,n);
for i=1:m                                                             
fi=X_train(i,:);
    for j=1:n
        for k=1:n
             if k==j
               pjj=(fi(j).*fi(j))./2;
               P(j,j)=pjj; 
             end
             if k~=j
               pkj=fi(j).*fi(k);
                P(k,j)=pkj;
             end
        end
     end
P=triu(P)';
a1=cell(1,n);
a1{1}=P(:,1);
   for c=2:n
     mi=P(:,c)';
     mi(1:c-1)=[];
     a1{c}=mi';
   end
a2 =vertcat(a1{1:n})'; 
a3=[a2 fi 1];
g(i,:)=a3;
end

%with qvec and dvec

% [m,d]=size(X_train);       
% %µ√µΩæÿ’Ûg
% g=zeros(m,2*d+1);
% F=zeros(d,d);
% for i=1:m
%     xi=X_train(i,:);
%     for j=1:d
%         for k=1:d
%             if k==j
%                 pjj=(xi(j).*xi(j))./2;
%                 F(j,j)=pjj;
% %             elseif j==k+1
% %                 pkj=xi(j).*xi(k);
% %                 F(k,j)=pkj;
% %             elseif k==j+1
% %                 pkj=xi(j).*xi(k);
% %                 F(k,j)=pkj;    
%             end
%         end
%     end
% a2=diag(F)';
% % aa1=diag(F);
% % aa2=diag(F,1);
% % F=triu(F)';
% % a1=cell(1,d);
% % a1{1}=F(:,1);
% %    for c=2:d
% %      mi=F(:,c)';
% %      mi(1:c-1)=[];
% %      a1{c}=mi';
% %    end
% % a2 =vertcat(a1{1:d})';
% % a2=[aa1;aa2]';
% a3=[a2 xi 1];
% g(i,:)=a3;
% end
