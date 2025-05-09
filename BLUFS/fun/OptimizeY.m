function [Y]= OptimizeY(S,W,Y_current,tau_3,alpha,X)
i_y=100
%% update Y 
    opts = [];  % no predefined parameter

    opts.X = Y_current; % specific the initial point
    opts.info_warning = 0; % display costomized warning messages
    opts.stepsize.type = 'ABB'; % set stepsize as alternating Barzilai-Borwein stepsize
    opts.stepsize.max_stepsize = 1000; % specify the maximum stepsize to improve the robustness
    opts.stepsize.min_stepsize = 0; % specify the minimum stepsize to improve the robustness
    opts.stepsize.init_stepsize = 1e-2; % specify the initial stepsize
    opts.gtol = 1e-8; % set the stopping criteria
    opts.local_info = 0; % set the display mode
    opts.postprocess = 1; % turn on the post-process 
    opts.linesearch = 0; % no linsearch
    opts.maxit = i_y;
    opts.info = 0;
    [Out] = stop_pencf(@funch1,opts,S,W,tau_3,alpha,Y_current,X);
    Y = Out.X;        % update X

    function [h_loss, h_grad] = funch1(Y,S,W,tau_3,alpha,Y_current,X)
%% Calculate the gradient and function values of h   
        h_grad  = -2*(X'*W-Y)-2*alpha*S*Y+2*tau_3*(Y-Y_current);
        h_loss  = (-1)*alpha*trace(Y'*S*Y) + (norm(X'*W-Y,'fro')^2)+tau_3*(norm(Y-Y_current,'fro')^2);
                  
    end






end