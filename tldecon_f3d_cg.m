function [IOut,paramsout]= tldecon_f3d_cg(I,noisy_Im,psfs,paramsin,I_diff,Code_mtx)

% Three Dimensional Full Fixed Sparsifying Transform regularized deconvolution 

%Inputs: 1) I: Reference spectral images (one under the other)
%        2) noisy_Im: Noisy measurements
%        3) paramsin: Structure that contains the input parameters of the simulation. The various fields are as follows -
%                   - nu: Weight on the data fidelity term in the problem formulation
%                   - num: Number of iterations of the algorithm
%                   - s: This is a vector of the same length as the number of algorithm iterations, and contains the respective 
%                        sparsity fractions (i.e., fraction of non-zeros in the sparse code matrix) to be used in the algorithm iterations.
%                   - ct: If set to 1, the code additionally outputs various performance metrics computed over the algorithm iterations. Otherwise, set to 0.
%                   - plot: If set to 1, the code additionally outputs PSNRs vs iterations plot. Otherwise, set to 0.
%                   - nl: If set to 1, it indicates that the input data is normalized (i.e., the peak intensity value in the reference image is 1). 
%                         For any other value of `nl', the code automatically applies a normalization before the algorithm begins.
%        4) psfs: (K x P) block matrix containing the PSFs (psfsize x psfsize) in each block

%Outputs:  1) IOut: Reconstructed images.
%          2) paramsout - Structure containing various outputs, and convergence or performance metrics 
%                         for the algorithm. Many of these are vectors (whose entries correspond to values at each iteration).
%                 - PSNRs: PSNR of each reconstruction at each iteration of the algorithm (output only when ct is set to 1)
%                 - PSNRs_diff: The same as PSNRs except that it is calculated with respect to the diffraction limited images
%                 - SSIMs: SSIM of each reconstruction at each iteration of the algorithm (output only when ct is set to 1)
%                 - SSIMs_diff: The same as SSIMs except that it is calculated with respect to the diffraction limited images
%                 - runtime: total execution time for the algorithm (output only when ct is set to 1)
%                 - itererror: norm of the difference between the reconstructions at successive iterations (output only when ct is set to 1)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Initializing algorithm parameters

%%OFK

%%OFK end
H = psfs;
psfsize=paramsin.psfsize;
[k,p]=size(psfs);
k=k/psfsize; % number of measurement planes
p=p/psfsize; % number of sources
K=k;
P=p;
[aa,bb]=size(I);
aa=aa/p; % aa is the size of the image ( assumes (aa x bb) image )

f = paramsin.f;
d = paramsin.d;
init = paramsin.init;
La2=(paramsin.nu)*(aa*bb*k);
num=paramsin.num;
et=paramsin.s;
ct=paramsin.ct;
nl=paramsin.nl;
my_plot=paramsin.plot;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if(nl~=1)
    for i=1:p
    y = indexer(I,i,1,aa);
    sc = (max(max(abs(y))));
    y = y/sc;
    I(1+(i-1)*aa:i*aa, 1:aa) = y;
    end
end

if(strcmp(init,'zeros'))
    Im = zeros(p*aa,aa);
    
elseif(strcmp(init,'meas'))
    for i=1:p
        y = indexer(noisy_Im,i,1,aa);
        sc = (max(max(abs(y))));
        sc = 1;
        y = y/sc;
        Im(1+(i-1)*aa:i*aa, 1:aa) = y; % to start the iterations with an initial image of intensity between [0 1]
    end

elseif(strcmp(init,'ls'))
    gamma_H = block_fft2(psfs, psfsize, aa);
    gamma_H = real(gamma_H);
    gamma_Hh = my_herm(gamma_H,aa);
    Y = block_fft2(noisy_Im,aa,aa);
    lambda = my_mul(gamma_H,gamma_Hh,aa);
%     Im = block_ifft2(my_mul(my_inv(lambda,p),my_mul(gamma_Hh,Y,aa),aa),aa);
%     Im = block_ifft2(my_mul(inv(lambda+1*eye(size(lambda))),my_mul(gamma_Hh,Y,aa),aa),aa);
% 	  Im = block_ifft2(my_mul(gamma_Hh,my_mul(my_inv(lambda,k),Y,aa),aa),aa);
	Im = block_ifft2(my_mul(gamma_Hh,my_mul(my_inv(lambda+5*kron(eye(k),ones(aa)),k),Y,aa),aa),aa);
    Im = real(Im);
elseif(strcmp(init,'mean'))
    Im = zeros(p*aa,aa);
    Im(1:aa, 1:aa) = indexer(noisy_Im,1,1,aa);
    Im(1+(p-1)*aa:p*aa, 1:aa) = indexer(noisy_Im,k,1,aa);
    for i= 2:p-1
        if f(i) < max(d) && f(i) > min(d)
            [~,j1] = min(abs(d-f(i)));
            if d(j1) < f(i)
                j2 = j1 + 1;
            else
                j2 = j1 - 1;
            end
            Im(1+(i-1)*aa:i*aa, 1:aa) = ((f(i)-d(j1))*indexer(noisy_Im,j2,1,aa) + (d(j2)-f(i))*indexer(noisy_Im,j1,1,aa))/(d(j2)-d(j1));
        elseif f(i) > max(d)
            Im(1+(i-1)*aa:i*aa, 1:aa) = indexer(noisy_Im,k,1,aa);
        elseif f(i) < min(d)
            Im(1+(i-1)*aa:i*aa, 1:aa) = indexer(noisy_Im,1,1,aa);
        end
    end
end

% Note that Im is our current estimate (nothing but the noisy measurements) 
% , and will be updated as the estimate at each iteration

if(ct==1)
    ittime = zeros(1,num);
    PSNRs = zeros(p,num);
    PSNRs_diff = zeros(p,num);
    SSIMs = zeros(p,num);
    SSIMs_diff = zeros(p,num);
    itererror = zeros(p,num);
end

for kp = 1:num
    kp;
    tic
    
    for i = 1:p
        Iiter(:,:,i) = indexer(Im,i,1,aa); % store Iiter to calculate itererror
    end
    
    C = sparse_coding(Im, aa, bb, p, et, num);
    tol = 1e-7;  max_it = 200;
    
    if(kp==1)
        gamma_H = block_fft2(H, psfsize, aa);
        gamma_H = real(gamma_H);
        gamma_ = my_mul(my_herm(gamma_H, aa), gamma_H, aa);
        f_noisy_Im = block_fft2(noisy_Im,aa,aa);
        ChHhy = La2*Code_mtx .* block_ifft2(my_mul(my_herm(gamma_H, aa), f_noisy_Im, aa),aa);
    end
    
    b = ChHhy + C;
    
    [Im, error, iterCG, flagCG] = cgconvtikPS_f3d(Im,max_it,tol,La2,b,Code_mtx,gamma_);
    error
    iterCG

    time=toc;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %Compute various performance or convergence metrics
    if(ct==1)
        ittime(kp)=time;
        
        for i=1:p
            Iim=indexer(Im,i,1,aa); Ii=indexer(I,i,1,aa); I_diffi=indexer(I_diff,i,1,aa);
            
            PSNRs(i,kp)=20*log10((sqrt(aa*bb))*1/norm(double(abs(Iim))-double(abs(Ii)),'fro'));
            PSNRs_diff(i,kp)=20*log10((sqrt(aa*bb))*1/norm(double(abs(Iim))-double(abs(I_diffi)),'fro'));
            SSIMs(i,kp)=ssim(Iim,Ii);
            SSIMs_diff(i,kp)=ssim(Iim,I_diffi);
            
            itererror(i,kp) = norm(squeeze(Iiter(:,:,i)) - Iim,'fro')/norm(Iim,'fro');
        end
    end
    
    %real time plot details
    if my_plot==1 && mod(kp,50)==0
        %close(figure(6));
        figure(6)
        clf
        for i=1:p
            plot(SSIMs(i,:));
            hold on;
        end
        
        bas = ['\nu=',num2str(La2),' s=',num2str(et(1))];
        x=' SSIM';
        son=[];
        for i=1:p
            son = [son, x, num2str(i), '=', num2str(SSIMs(i,kp))];
        end
        title({[bas],[son]});
        
        grid minor;
        
        Legend = cell(p,1);
        for i=1:p
            Legend{i}=['SSIM',num2str(i)];
        end
        legend(Legend,'Location','southeast')
        
        figure(7)
        
        subplot(1,2,1)
        imagesc(I);
        colormap gray
        axis image;
        colorbar
        title('Original')
        
        subplot(1,2,2)
        imagesc(Im);
        colormap gray
        axis image;
        colorbar
        title('Reconstructed')
        
        drawnow;
    end
    
%     stopping criterion
    if(min(itererror(:,kp)<2e-4)) || ( kp>400 && sum(  SSIMs(:,kp) > SSIMs(:,kp-1) ) == 0 )
        break
    end   
end

figure(6)
clf
for i=1:p
    plot(SSIMs(i,:));
    hold on;
end

bas = ['\nu=',num2str(La2),' s=',num2str(et(1))];
x=' SSIM';
son=[];
for i=1:p
    son = [son, x, num2str(i), '=', num2str(SSIMs(i,kp))];
end
title({[bas],[son]});

grid minor;

Legend = cell(p,1);
for i=1:p
    Legend{i}=['SSIM',num2str(i)];
end
legend(Legend,'Location','southeast')

figure(7)

subplot(1,2,1)
imagesc(I);
colormap gray
axis image;
colorbar
title('Original')

subplot(1,2,2)
imagesc(Im);
colormap gray
axis image;
colorbar
title('Reconstructed')

drawnow;

%Outputs
IOut = Im;

if(ct==1)
    paramsout.PSNRs_last=PSNRs(:,kp);
    paramsout.PSNRs_diff_last=PSNRs_diff(:,kp);
    paramsout.PSNRs=PSNRs;
    paramsout.PSNRs_diff=PSNRs_diff;
    paramsout.SSIMs_last=SSIMs(:,kp);
    paramsout.SSIMs_diff_last=SSIMs_diff(:,kp);
    paramsout.SSIMs=SSIMs;
    paramsout.SSIMs_diff=SSIMs_diff;
    paramsout.itererror=itererror;
    paramsout.runtime=sum(ittime);
end
