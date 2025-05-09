function [result,row,obj] = BLUFS (X,c,beta,lamda,alpha,mu,s)
%数据预处理
[d,n]     = size(X);
%对样本进行归一化处理
X        = NormalizeFea(full(X),1);
eps       = 1e-7;
maxiter   = 50;%循环次数
tol       = 1e-3;%收敛条件
k         = n      ;
tau_1     = 0.8    ;
tau_2     = 0      ;
tau_3     = 0      ;
 
%%
%构建S
D         = EuDist2(X');
optt      = mean(D(:)); 
options.NeighborMode = 'KNN';
options.WeightMode   = 'HeatKernel'; 
options.k = 10;  
options.t = optt; 
S         = constructW(X',options);%权重矩阵 S 中的元素 S(i,j) 表示数据点 i 和 j 之间的相似性或权重。

dd        = sqrt(1./max(sum(S,2),eps)); %sum(S,2)对S的行求和，生成一个列向量，然后逐元素求倒平方，用于对S进行归一化
NS        = sparse(diag(dd)*S*diag(dd));%稀疏S
%%
%初始化W
 W_current = random('Normal', 0, 1, [d, c]); % 生成一个 m 行 n 列的随机矩阵

%%
%初始化P矩阵
[P_current] = initialiseP(X);
LL        = diag(sum(P_current)) - P_current;
LP        = (LL + LL') / 2;

%%
% 生成一个矩阵Y，满足列正交,c作为分类数
F = zeros(n,c);
for i = 1:n
    idxr=randi(c);
    F(i,idxr) = 1;

end
FFt = F' * F;
% 求逆
FFt_inv = inv(FFt);
% 取平方根
FFt_inv_sqrt = sqrt(FFt_inv);
% 乘以 F
 Y_current= F * FFt_inv_sqrt;
%%

for iter = 1:maxiter
    W_old  = W_current; 
 W_current = OptimizeW(X,Y_current,lamda,beta,LP,W_current,tau_2);
   P_current = OPTIMIZEP(W_current,X,P_current,k,mu,tau_1);
  % P_current = HH(X', 3)高斯核函数
 LL = diag(sum(P_current)) - P_current;
 LP = (LL + LL') / 2;
 Y_current = OptimizeY(NS,W_current,Y_current,tau_3,alpha,X);
 
 %%    检查收敛
A=W_current'*X;
parfor i = 1:n
    for j = 1:n
        diff = A(:, i) - A(:, j); % Difference between projected points
        term1(i,j) = beta * norm(diff)^2 * P_current(i, j); % First term in the sum
    end
end
term2   =  norm(term1,'fro')^2 + mu * norm(P_current, 'fro')^2;
term3   =  norm(X'*W_current - Y_current, 'fro')^2;
term4   =  lamda*norm(W_current, 'fro')^2;
term5   =  -alpha*trace(Y_current'*NS*Y_current);
objFunc = term3 + term4+term5 + term2  ;
    obj(iter) = objFunc;
    if iter>1
        error_obj(iter) = abs(obj(iter)-obj(iter-1))/(1+abs(obj(iter-1)));      % error_obj
        error_W(iter) = norm(W_current-W_old)^2/(1+norm(W_old)^2);
        fprintf('%5d\t  %6.2e\t %6.2e\t %6.2e\t %6.2e\n ',iter, obj(iter), error_obj(iter),error_W(iter));
          if  ((error_obj(iter) <tol)); break; end
          if obj(iter)==0; break; end
        if iter == maxiter; fprintf('The number of iterations reaches maxiter.\n'); end
    end
end
row_norms = vecnorm(W_current, 2, 2);        % 计算每一行的 l2 范数
[~, idx] = sort(row_norms, 'descend');       % 找到范数最大的前 s 行，按行范数降序排列
W_new = zeros(size(W_current));              % 初始化稀疏矩阵
W_new(idx(1:s), :) = W_current(idx(1:s), :); % 仅保留前 s 行
W_current = W_new;                           % 更新 W
%构造特征选择结果 
[row, ~] = find(W_current(:,1)); 
result = zeros(s,n);

for i = 1:s
    result(i,:) = X(row(i,:),:);
end
end

