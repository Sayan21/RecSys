clear all;
Data=load('~/Downloads/Yandex/user_url_test_1M_14days_5Clicks.txt');
Data(:,1)=Data(:,1) + 1; Data(:,2)=Data(:,2)+1;

nD = max(Data(:,1));
nV = max(Data(:,2));
val = ones(size(Data,1),1);
X=sparse(Data(:,1),Data(:,2),val,nD,nV);
X=logical(X);
X=double(X);
clearvars Data;

% AUC(K =100)=> BPR: 0.655 PLSI: 0.573

Xsum=full(sum(X,2)); n=unique(Xsum);
f=zeros(max(n)-min(n)+1,1);
for i=min(n):1:max(n)
    f(i)=length(find(Xsum==i));
end
    
figure; bar(f); xlim([0 60]);

t0=tic;
M1 = full(sum(X,1)');
Z1=sum(M1); M1=M1./Z1;


M2=X'*X;
Z2=sum(sum(M2,1),2); M2=M2./Z2;

K =100; %Number of Clusters
[U,S]=eigs(M2,K);
s = diag(S);
W = U*diag(1./sqrt(s));
clearvars M2 U S;

Xsum = full(sum(X,2));
Z3 = sum(Xsum.*Xsum.*Xsum);
Mx=X*W./power(Z3,1/3);

% T2 = zeros(K,K,nD);
% for i=1:nD
%     T2(:,:,i) = Mx(i,:)'*Mx(i,:);
% end
% T2=tmprod(T2,Mx',3);

G = zeros(K,K,K);
for i=1:K
    for j=1:K
        G(:,i,j)=(Mx(:,i).*Mx(:,j))'*Mx;
    end
    sprintf('Matrix Multiplied: %d',i)
end

% Extract tensor eigenvalues
G=tensor(G);
eigvals = zeros(K,1);
V=zeros(K,K);
for k=1:K
    G=symmetrize(G);
    [s,U]=sshopm(G,'Tol',1e-16);
    eigvals(k)=s; V(:,k)=U;

    G=G-tensor(ktensor(s,U,U,U));
    sprintf('EigenValue Extracted: %d',k)
end

%W2 = W*inv(W'*W);
clearvars G;
V2 = zeros(nV,K); W2 = pinv(W');
for k=1:K
    V2(:,k)=eigvals(k)*W2*V(:,k);
end
P = zeros(nV,K);
for k=1:K
    if eigvals(k)>0
        P(:,k)=V2(:,k);
    else
        P(:,k) = -V2(:,k);
    end

    sel=find(P(:,k)<0);
    P(sel,k)=0;
    P(:,k) = P(:,k)./sum(P(:,k));
end




%P=normalize_cols(V2);
beta = .001; P = P + beta*ones(size(P))/nV;
for k=1:K
    P(:,k)=P(:,k)./sum(P(:,k));
end

% TRec = tensor(ktensor(eigvals(1),V(:,1),V(:,1),V(:,1)));
% for k=2:K
%     TRec = TRec + tensor(ktensor(eigvals(k),V(:,k),V(:,k),V(:,k)));
% end

pi = 1./(eigvals.*eigvals);
pi = pi/sum(pi);

Pu = zeros(K,nD);
for u=1:nD
    [i j n]=find(X(u,:));
    L = P(j',:);
    Lprob = log(pi) + sum(log(L),1)';
    [~,imax]=max(Lprob);
    ll = Lprob-Lprob(imax);
    prob = exp(ll) + realmin('double');
    prob = prob/sum(prob);
    if ~isempty(find(isnan(prob),1))
        error('NaN in user probability');
    end
    Pu(:,u)=prob;
    if(mod(u,1000)==0) 
        fprintf(1,'User Probability Extracted: %d\n',u); toc(t0);
    end
end
  

t1=cputime;


filename = sprintf('~/Downloads/Yandex/urlprob_search_K%d_1M_14_5Clicks.txt',K);
dlmwrite(filename,P','delimiter',' ');

filename = sprintf('~/Downloads/Yandex/topicprob_search_K%d_1M_14_5Clicks.txt',K);
dlmwrite(filename,eigvals,'delimiter',' ');
filename = sprintf('~/Downloads/Yandex/userprob_search_K%d_1M_14_5Clicks.txt',K);
dlmwrite(filename,Pu,'delimiter',' ');

    
clearvars X;
Data = load('~/Downloads/Yandex/user_url_test_1M_14days_5Clicks.txt');
N = size(Data,1); 
Data(:,1)=Data(:,1) + 1; Data(:,2)=Data(:,2)+1;

M = [5,10,20,40,60,80,100,200,300,400,500];
sumAP=zeros(length(M),1); count=0; sumPrec = zeros(length(M),1); sumRecall=zeros(length(M),1);
uAP = zeros(3000,1);
for u=1:nD
    for l=1:length(M)
       Py_u = P*Pu(:,u);
       sel=(Data(:,1)==u);
       [score,ID]=sort(Py_u,'descend');

       if ~isempty(find(sel==1, 1))
            AP=averagePrecisionAtK(Data(sel,2),ID(1:M(l)),M(l));
            sumAP(l) = sumAP(l) + AP;

            prec = length( intersect(Data(sel,2),ID(1:M(l))) )/M(l);
            sumPrec(l) = sumPrec(l)+prec;

            recall = length( intersect(Data(sel,2),ID(1:M(l))) )/length(Data(sel,2));
            sumRecall(l) = sumRecall(l) + recall;

            count = count+1;
            
            if(M(l)==500)
                c=count/length(M);
                uAP(c,1) = length(Data(sel,2)); uAP(c,2)=AP;
            end
       end
       if(M(l)==500)
           c = count/length(M);
           sprintf('%d: MAP:%f Precision:%f Recall:%f',c,sumAP(l)/c,sumPrec(l)/c,sumRecall(l)/c)
           
       end
    end
end
