function[ W ] = OptimizeW(X,Y,lambda,beta,LP,W_current,tau_2)
[d, n] = size(X);

AA = (tau_2+lambda)*eye(d);
BB = beta*X*LP*X';
CC = X*X';
DD = X*Y+tau_2*W_current;
A = AA+BB+CC;
W = A\DD;
