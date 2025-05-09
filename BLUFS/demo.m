clc
clear 
load COIL20.mat          
i_1=1               ;
X  =X'              ;
[d,n] = size(X)     ;
c     =  10         ;   %谱聚类设置的标签数
lamda =  1          ;
beta  =  1          ;
alpha =  1          ;
mu    =  1          ;
ACCmatrix = zeros(10, 1);  %ACC评价结果矩阵
for s = 10:10:100
  [~,row,obj] = BLUFS(X,c,beta,lamda,alpha,mu,s);%output2定义为特征提取的特征子集
  for i = 1:s
      result(i,:) = X(row(i),:);
  end
    output2=result';
  for m = 1:1:10
      k=20 ;
      [idx, C] = litekmeans(output2, k,'MaxIter', 50,'Replicates',10);%k-means聚类
      idx1 = bestMap(Y,idx)  ;         %标准化
      [ACC,maxtric]=Eva_CA(idx1,Y); %ACC评价 ,maxtric是混淆矩阵                            
      nmi = MutualInfo(idx1,Y);
      ACCmatrix(m,1)=ACC;
      nmimatrix(m,1)=nmi;
   end
    NMI=mean(nmimatrix(:,1));
    ACC = mean(ACCmatrix(:, 1));
    matrix(i_1, 1) = ACC;  %matrix提取每个不同特征数所对应的ACC
    std_dev(i_1,1)=std(ACCmatrix);
    matrix(i_1,2)  = NMI;
    std_dev(i_1,2)=std(nmimatrix);
    i_1=i_1+1;
end








