%%

% There are 3 parameter optimzation scripts present in this m file regarding
% the function tldecon_g2. These correspond to 20,25, and 30 dB SNR optimizations 
% as can be adjusted on by "snrn" parameter.

% 1. Parameter intervals to be scanned are entered in "FIRST/SECOND/THIRD 
% OPIMIZATION INPUTS" sections (Don't forget to specify the "outpath"s 
% seperately that the optimization results are going to be saved.)

% 2. Then, run the whole script by pressing F5.

% 3. After all reconstructions corresponding to all given parameters are
% done, the "rmse cube" covering all parameter combinations are automatically
% saved to the "outpath" and the optimal parameter group can be found by manually 
% making contour plots in Matlab with these automaticly created "rmse" files.

%% FIRST OPIMIZATION INPUTS

imsel = 1; % 0 for SI images, 1 for indian pines
aa=128; %image size
factor=1/4; %image resize factor
snrn = 30; %SNR of the measurements
add_noise = 0; % 1 for noisy, 0 for noiseless
k = 3; %num of meas planes
p = 6; %num of sources
% dn = [0.00    0.50    1.00]; % fill this with normalized values
dn = [0.10    0.50    0.90]; % fill this with normalized values
color = 1; % 1 for colored; 0 for uncolored
psdesign = 2; % 1 or 2

measurement_simulation; %input image set and psf initailizations
% Algorithm Parameters
paramsin.num = 2000; 
paramsin.numiter = 1; 
paramsin.n = 36;
paramsin.n1 = 36;
paramsin.n2 = 3; %transform size in wavelength dimension
paramsin.N = aa*aa*p;
paramsin.lambda0 = 0.001;
paramsin.psfsize = psfsize;
paramsin.init = 'zeros';
paramsin.W01 = kron(dctmtx(6),dctmtx(6));
paramsin.W3d = kron(dctmtx(paramsin.n2),kron(dctmtx(6),dctmtx(6)));
paramsin.ct = 1;
paramsin.plot = 0;
paramsin.d = d; % normalized measurement plane distances 
paramsin.f = f; % focal distances of sources
paramsin.co = 0;
paramsin.nl = 1;
% load('learned_transform3.mat');
% paramsin.W01=learned_transform3;
paramsin.num_first = 1; %IS iteration number

% output folder path:
outpath = '.\objects\ball128\g3dc_iu\meas[0.10_0.50_0.90]\colored_mask\noiseless';
mkdir(outpath);

% parameter set to be scanned:
noise_real = 1; %number of noise realizations
nu_array = [1e4 2e4 5e4 1e5 2e5 5e5]; %nu values to optimize
s_array = [0.065 0.075 0.085] ; %s values to optimize

% paramsin: Structure that contains the input parameters of the simulation. The various fields are as follows -
%                   - nu: Weight on the data fidelity term in the problem formulation
%                   - num: Number of iterations of the algorithm
%                   - s: This is a vector of the same length as the number of algorithm iterations, and contains the respective 
%                        sparsity fractions (i.e., fraction of non-zeros in the sparse code matrix) to be used in the algorithm iterations.
%                   - ct: If set to 1, the code additionally outputs various performance metrics computed over the algorithm iterations. Otherwise, set to 0.
%                   - plot: If set to 1, the code additionally outputs PSNRs vs iterations plot. Otherwise, set to 0.
%                   - nl: If set to 1, it indicates that the input data is normalized (i.e., the peak intensity value in the reference image is 1). 
%                         For any other value of `nl', the code automatically applies a normalization before the algorithm begins.

% FIRST PARAMETER OPTIMIZATION (do not change)

sizeNU = size(nu_array,2);
sizeS = size(s_array,2);
counter = 1;

rmse30 = NaN(p,sizeNU,sizeS,noise_real);
SSIM30 = NaN(p,sizeNU,sizeS,noise_real);
rmse30_diff = NaN(p,sizeNU,sizeS,noise_real);
SSIM30_diff = NaN(p,sizeNU,sizeS,noise_real);
loading = 'Loading %';

for n=1:noise_real
    measurement_simulation;
    iternum = 1;
    for kk=1:sizeNU
        paramsin.nu = nu_array(kk)/(aa*aa);
        for l=1:sizeS
            paramsin.s = s_array(l)*ones(1,2000);
            [IOut,paramsout] = tldecon_g3dc_iu(I,noisy_Im_C,psfs,paramsin,I_diff,Code_mtx);
            
            for ii = 1:p
                rmse30(ii,kk,l,n) = 1/aa*norm(indexer(IOut,ii,1,aa)-indexer(I,ii,1,aa),'fro');
                SSIM30(ii,kk,l,n) = ssim(indexer(IOut,ii,1,aa),indexer(I,ii,1,aa));
                rmse30_diff(ii,kk,l,n) = 1/aa*norm(indexer(IOut,ii,1,aa)-indexer(I_diff,ii,1,aa),'fro');
                SSIM30_diff(ii,kk,l,n) = ssim(indexer(IOut,ii,1,aa),indexer(I_diff,ii,1,aa));
            end
            
            disp(loading); disp([counter*100/(noise_real*sizeNU*sizeS) kk l n mean(SSIM30_diff(:,kk,l,n))]);
            
            figure(6)
            clf
            for i=1:p
                plot(paramsout.SSIMs_diff(i,:));
                hold on;
            end
            
            bas = ['\nu=',num2str(nu_array(kk)),' s=',num2str(s_array(l))];
            x=' SSIM';
            son=[];
            for i=1:p
                son = [son, x, num2str(i), '=', num2str(paramsout.SSIMs_diff_last(i))];
            end
            title({[bas],[son]});
            
            grid minor;
            
            Legend = cell(p,1);
            for i=1:p
                Legend{i}=['SSIM',num2str(i)];
            end
            legend(Legend,'Location','southeast')
                      
            saveas(figure(6),[outpath, '\','NR_',num2str(n),'_',num2str(iternum)],'png');
            iternum = iternum + 1;
            counter = counter + 1;
        end
    end
end

save([outpath '\paramsin.mat'], 'paramsin');
save([outpath '\paramsout.mat'], 'paramsout');
save([outpath '\IOut.mat'], 'IOut');
save([outpath '\rmse30.mat'], 'rmse30');
save([outpath '\SSIM30.mat'], 'SSIM30');
save([outpath '\rmse30_diff.mat'], 'rmse30_diff');
save([outpath '\SSIM30_diff.mat'], 'SSIM30_diff');
copyfile('tldecon_g_opt.m',outpath,'f');
copyfile('tldecon_g3dc_iu.m',outpath,'f');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% close all;
% clear all;

