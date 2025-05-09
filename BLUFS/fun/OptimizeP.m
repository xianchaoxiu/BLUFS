
%加上惩罚项
function [P] = OPTIMIZEP(W, X, P_prev, k,mu, tau)


[n, ~] = size(X'); % Number of samples
P = zeros(n, n); % Initialize similarity matrix

%%
% 构建eta
eta = zeros(n, n); 
A = W'*X;
parfor i = 1:n
    for j = 1:n
        eta(i, j) = norm(A(:, i) - A(:, j))^2;
    end
end
%%
%对eta进行排序，值最小的排名靠前，以提取最近邻
for i = 1:n
const1=1/(2*mu);
const2=-tau/(mu+tau);
[sorted_eta, idx] = sort(eta(:,i), 'ascend');
term1=const1*sorted_eta(1:k,:);
term2=const2*P_prev(idx(1:k,:),i);
rho=(1/k)*(term1+term2);
rho=sum(rho,"all");
   %  [sorted_eta, idx] = sort(eta(:,i), 'ascend');
   %  获得rho
   %  sum_eta_k = sum(eta(1:k,i));
   %  rho = (1 / k) + (1/k*2*mu)*(sum(-sorted_eta(1:k)))+(tau/k*(mu+tau))*(sum(P_prev(idx(1:k)),i))
   %  rho=1/k;
   %  for j = 1:k
   %  term1 = sorted_eta(j) / (2 * mu); % 分量1: -ηij/(2μ)
   %  term2 = -(tau / (mu + tau)) * P_prev(idx(j),i); % 分量2: τ1/(μ + τ1) * P^k_ij
   %  rho = rho + (1 / k) * (term1 + term2) % 累加至 rho
   %  end
   %  更新 P_i
    for j = 1:k
        P(idx(j),i ) = max(-sorted_eta(j) / (2 * mu) +  (tau / (mu + tau)) * P_prev(idx(j),i) + rho, 0);
                         
    end
end
end
