function P = HH(X, sigma)
    % 输入：
    %   X - 数据矩阵，每一行是一个样本
    %   sigma - 高斯核的带宽参数
    % 输出：
    %   P - 归一化的高斯核相似性矩阵

    % 计算样本之间的欧几里得距离
    dist_matrix = squareform(pdist(X, 'euclidean'));
    
    % 计算高斯核权重
    P = exp(-dist_matrix.^2 / (2 * sigma^2));
    
    % 归一化每一行
    row_sums = sum(P, 2);
    P = bsxfun(@rdivide, P, row_sums);
end