function [X, error, iter, flag] = cgconvtikPS_f3d(X0,max_it,tol,nu,b,Code_mtx,gamma_)
% [X, error, iter, flag] = cgconvtikPS(S,K,H, Wc,d1,d2,Wd,X0,Y,lambda,max_it,tol)
%
% S     : Number of monochromatic sources, each with a different wavelength
% K     : Number of measurement planes, each with a different distance from PS
% H     : 2-D (P1 X P2) blurring kernel
% Wc    : (L1 X L2) Operator weight *image* (corresp to noise strength at each pixel)
%         (could also be written as a diagonal weighting matrix of size
%         (L1*L2)x(L1*L2) to be multiplied with the vectorized data)
% d1    : 2-D derivative operator along 1st coordinate direction (col vec)
%         e.g. d1 = [-1 0 1]';
% d2    : 2-D derivative operator along 2nd coordinate direction (row vec)
%         e.g. d2 = [-1 0 1];
% Wd    : (2L1 X L2) Regularizer weight *image*
%         Note: Wd is essentially two images concatenated vertically:
%         Wd = [Wd1;Wd2] corresponding to the weights for two derivative kernels
%         (could also be written as a diagonal weighting matrix of size
%         (2L1*L2)x(L1*L2))
% X0    : (L1 X L2) Initial guess image
% Y     : (L1 X L2) Data image
% M     : (L1*L2 X L1*L2) Preconditioner matrix. Generally diagonal.
%         (OMITTED TO AVOID SLOWING DOWN OF THE ALG.)
% lambda : Sqrt of regularization parameter (i.e. Reg. par. =lambda^2)
% max_it: Maximum number of iterations
% tol   : Error tolerance
%
% X     : (L1 X L2) Solution image
% error : Corresponding error norm
% iter  : Number of iterations performed
% flag  : Convergence flag 0 = solution found to tolerance
%                          1 = no convergence given max_it
%
% Solves the Tikhonov regularized problem:
%
% ||y - Hx||^2_Wc + lambda^2 ||Dx||^2_Wd  ( where ||x||^2_W = (x)'*W*(x) )
%
% corresponding to image deblurring using the conjugate gradient alg. 
% Huge matrix-vector multiplications are avoided using calls to conv2.m.
% Specialized to the case that y corresponds to measurements of a photon 
% sieve (PS) system and x corresponds to source images at different wavelengths.
%
% Based on cg.m (ftp netlib2.cs.utk.edu; cd linalg; get templates.ps), and 
% based on cgconvtik.m (W. C. Karl SC717)

% NONNEGATIVITY constraint could also be imposed

% PRECONDITIONING IS OMITTED

%% Sizes of the matrices
L1 = size(X0,2);
bnrm2 = norm( b ,'fro');

%% This computes (mu*C'H'HC + I) * x0

Tmp1 = Code_mtx .* block_ifft2(my_mul(gamma_,block_fft2(Code_mtx .* X0,L1,L1),L1),L1);
Tmp1 = nu * Tmp1;
Tmp1 = Tmp1 + X0;

clear B Bweighted

%% This finds r = b - Ax;
% r = b - Tmp1(:) - lambda^2*Tmp2(:);
r = b - Tmp1;
X = X0;
flag = 0;

% clear Tmp1 Tmp2;
clear Tmp1;

error = norm( r ,'fro') / bnrm2;
if ( error < tol ); tol = tol/10; end

%% Begin iteration
for iter = 1:max_it
%     iter
    z =r;    %M \ r;  PRECONDITIONING IS OMITTED
    rho = sum(sum(r.*z));
    if ( iter > 1 ) % direction vector
        beta = rho / rho_1;
        p = z + beta*p;
    else
        p = z;
    end
    
    % q = A*p; --> q = (C'*Wc*C + lambda^2 D'*Wd*D)p
    
    %% This computes C'*Wc*C* p
    
    Tmp1 = Code_mtx .* block_ifft2(my_mul(gamma_,block_fft2(Code_mtx .* p,L1,L1),L1),L1);
    Tmp1 = nu * Tmp1;
    Tmp1 = Tmp1 + p;
    
    %% This computes d1'*Wd1*d1*p + d2'*Wd2*d2*p
    
    
    clear P;
    
    % This finds q = A*p;
%     q = Tmp1(:) + lambda^2*Tmp2(:);
    q = Tmp1;
    
%     clear Tmp1 Tmp2;
    clear Tmp1;
    
    alpha = rho / sum(sum(p.*q));
    X = X + alpha * p;          % update approximation vector
    r = r - alpha*q;            % compute residual
    error = norm( r ,'fro') / bnrm2;  % check convergence
    
    if ( error <= tol ); break, end
    rho_1 = rho;
%     error
end

if ( error > tol ); flag = 1; end % no convergence

% END cgconvtik.m