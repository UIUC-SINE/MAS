% by Fatih Çaðatay Akyön & Ulaþ Kamacý 2017
% fatihcagatayakyon@gmail.com & kamaci.ulas@gmail.com
%% tldecon_g tester

% paramsin: Structure that contains the input parameters of the simulation. The various fields are as follows -
%                   - nu: Weight on the data fidelity term in the problem formulation
%                   - num: Number of iterations of the algorithm
%                   - numiter: Number of iterations within transform learning (i.e., iterations of sparse coding and transform update, with fixed image)
%                   - num_first: Number of iterations of (image update + sparse coding) (i.e., iterations of sparse coding and transform update, with fixed transform)
%                   - n: Patch size, i.e., Total number of pixels in a square patch
%                   - N: Number of training signals used in the transform learning step of the algorithm.
%                   - W01: initial transform for the algorithm
%                   - r: Patch Overlap Stride (this implementation works with r=1)
%                   - s: This is a vector of the same length as the number of algorithm iterations, and contains the respective 
%                        sparsity fractions (i.e., fraction of non-zeros in the sparse code matrix) to be used in the algorithm iterations.
%                   - ct: If set to 1, the code additionally outputs various performance metrics computed over the algorithm iterations. Otherwise, set to 0.
%                   - co: If set to 1, the code additionally outputs various algorithm convergence metrics computed over the algorithm iterations. Otherwise, set to 0.
%                   - nl: If set to 1, it indicates that the input data is normalized (i.e., the peak intensity value in the reference image is 1). 
%                         For any other value of `nl', the code automatically applies a normalization before the algorithm begins.
%% INPUTS
imsel = 1; % 0 for SI images, 1 for indian pines
aa = 128; %image size
factor = 1/4; %image resize factor
snrn = 30; %SNR of the measurements
add_noise = 0; % 1 for noisy, 0 for noiseless
k = 3; %num of meas planes
p = 6; %num of sources
% dn = [0.00    0.50    1.00]; % fill this with normalized values
dn = [0.10    0.50    0.90]; % fill this with normalized values
% dn = [0.00    0.20    0.40    0.60    0.80    1.00]; % fill this with normalized values
color = 1; % 1 for colored; 0 for uncolored
psdesign = 1;

measurement_simulation; %input image set and psf initailizations
% Algorithm Parameters
paramsin.nu = 100000/(aa*aa*k); 
paramsin.num = 1000;
paramsin.numiter = 1; 
paramsin.n = 36;
paramsin.n1 = 36;
paramsin.n2 = 3; %transform size in wavelength dimension
paramsin.N = aa*aa*p;
paramsin.s = 0.065*ones(1,5000);
paramsin.lambda0 = 0.001;
paramsin.psfsize = psfsize;
paramsin.init = 'zeros';
% paramsin.W01 = D;
% paramsin.W02 = D;
% paramsin.W01 = TV_mtx(36);
% paramsin.W02 = TV_mtx(36);
paramsin.W01 = kron(dctmtx(6),dctmtx(6));
paramsin.W3d = kron(dctmtx(paramsin.n2),kron(dctmtx(6),dctmtx(6)));
% paramsin.W3d = paramsin.W3d + randn(size(paramsin.W3d));
% paramsin.W02 = kron(dctmtx(6),dctmtx(6));
% paramsin.W01 = eye(36);
% paramsin.W02 = eye(36);
% paramsin.W01 = 0.02*randn(36,36);
% paramsin.W02 = 0.02*randn(36,36);
% paramsin.W01=sqrt(diag(1./diag(paramsin.W01*paramsin.W01')))*paramsin.W01;
% paramsin.W02=sqrt(diag(1./diag(paramsin.W02*paramsin.W02')))*paramsin.W02;
paramsin.ct = 1;
paramsin.plot = 1;
paramsin.d = d; % normalized measurement plane distances 
paramsin.f = f; % focal distances of sources
paramsin.co = 0;
paramsin.nl = 1;
% load('learned_transform3.mat');
% paramsin.W01=learned_transform3;
paramsin.num_first = 1; %IS iteration number

%% RUN TLDECON_G
% [IOut,paramsout] = tldecon_g3d(I,xxx,psfs,paramsin,I_diff);
% [IOut,paramsout] = tldecon_mf1(I,xxx,psfs,paramsin,I_diff);
% [IOut,paramsout] = tldecon_g2(I,noisy_Im,psfs,paramsin);
% [IOut,paramsout] = tldecon_mf1c(I,noisy_Im_C,psfs,paramsin,I_diff,Code_mtx);
[IOut,paramsout] = tldecon_g3dc_iu(I,noisy_Im_C,psfs,paramsin,I_diff,Code_mtx);

mean(paramsout.SSIMs_diff_last)
% for j=1:paramsin.n2
% figure(9+j)
% for i=1:36
% subplot(6,6,i);
% tiro = reshape(paramsout.transform(i,:),6,6,paramsin.n2);
% imagesc(tiro(:,:,j));
% set(gca,'XTick',[]);
% set(gca,'YTick',[]);
% end
% colormap gray
% end

% IOut1=indexer(IOut,1,1,aa); IOut2=indexer(IOut,2,1,aa); IOut3=indexer(IOut,3,1,aa);
% DL_PSNR01=20*log10((sqrt(aa*aa))*1/norm(double(abs(IOut1))-double(abs(I11)),'fro'));
% DL_PSNR02=20*log10((sqrt(aa*aa))*1/norm(double(abs(IOut2))-double(abs(I22)),'fro'));
% DL_PSNR03=20*log10((sqrt(aa*aa))*1/norm(double(abs(IOut3))-double(abs(I33)),'fro'));

%I: orignal images
%noisy_Im: noisy measurements
%psfs: all psfs
