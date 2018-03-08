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

%image related:
snrn = 30; %SNR of the measurements

imsel = 1; % 0 for SI images, 1 for indian pines
k = 4; %num of meas planes
p = 5; %num of sources
dn = [0.1493    0.2575    0.6507    0.9543]; % fill this with normalized values

measurement_simulation;
load('pine145_10_noisy_4rand_meas_30SNR');

% Algorithm Parameters
paramsin.num = 100; % number of iterations
paramsin.psfsize = psfsize; 
paramsin.ct = 1; % computes PSNRs, itererror, and runtime if 1
paramsin.nl = 1;
paramsin.plot = 0;
paramsin.d = d; % normalized measurement plane distances 
paramsin.f = f; % focal distances of sources
paramsin.init = 'zeros'; % 'zeros' for zeros initialization
                % 'ls' for least squares solution initialization
                % 'mean' for assigning mean of the measure.ments to the
                % remaining sources
                % 'meas' for k=5, p=5 scenario.

% output folder path:
outpath = '.\full_images\uncoded\meas[0.1493_0.2575_0.6507_0.9543]\Results_30SNR_zeros_init';
mkdir(outpath);

% parameter set to be scanned:
noise_real = 10; %number of noise realizations
nu_array = [55]; %nu values to optimize
s_array = [0.010]; %s values to optimize

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

runtime = zeros(1,noise_real);
rmse30 = NaN(p,sizeNU,sizeS,noise_real);
PSNR30 = NaN(p,noise_real);
PSNR30_diff = NaN(p,noise_real);
SSIM30 = NaN(p,noise_real);
SSIM30_diff = NaN(p,noise_real);
IOuts = zeros(p*aa,aa,noise_real);
loading = 'Loading %';

for n=1:noise_real
    measurement_simulation;
    iternum = 1;
    for k=1:sizeNU
        paramsin.nu = nu_array(k)/(aa*aa);
        for l=1:sizeS
            paramsin.s = s_array(l)*ones(1,1000);
            [IOut,paramsout] = tldecon_mf1(I,noisy_Ims(:,:,n),psfs,paramsin,I_diff);
            
            IOuts(:,:,n) = IOut;
            runtime(n) = paramsout.runtime;
            PSNR30(:,n) = paramsout.PSNRs_last;
            PSNR30_diff(:,n) = paramsout.PSNRs_diff_last;
            SSIM30(:,n) = paramsout.SSIMs_last;
            SSIM30_diff(:,n) = paramsout.SSIMs_diff_last;
            for i = 1:p
                rmse30(i,k,l,n) = 1/aa*norm(indexer(IOut,i,1,aa)-indexer(I_diff,i,1,aa),'fro');
            end
            
            disp(loading); disp([counter*100/(noise_real*sizeNU*sizeS) k l n mean(SSIM30_diff(:,n))]);
            
            figure(6)
            clf
            for i=1:p
                plot(paramsout.PSNRs_diff(i,:));
                hold on;
            end
            
            bas = ['\nu=',num2str(nu_array(k)),' s=',num2str(s_array(l))];
            x=' PSNR';
            son=[];
            for i=1:p
                son = [son, x, num2str(i), '=', num2str(paramsout.PSNRs_diff_last(i))];
            end
            title({[bas],[son]});
            
            grid minor;
            
            Legend = cell(p,1);
            for i=1:p
                Legend{i}=['PSNR',num2str(i)];
            end
            legend(Legend,'Location','southeast')
                      
            saveas(figure(6),[outpath, '\','NR=',num2str(n),'_',num2str(iternum)],'png');
            saveas(figure(6),[outpath, '\','NR=',num2str(n),'_',num2str(iternum)],'fig');
            iternum = iternum + 1;
            counter = counter + 1;
        end
    end
end

snr30.runtime = runtime;
snr30.runtime_av = mean(runtime);
snr30.PSNR30 = PSNR30;
snr30.PSNR30_diff = PSNR30_diff;
snr30.PSNR30_av = mean(PSNR30,2);
snr30.PSNR30_diff_av = mean(PSNR30_diff,2);
snr30.PSNR30_av_over_images = mean(PSNR30);
snr30.PSNR30_diff_av_over_images = mean(PSNR30_diff);
snr30.PSNR30_av_av = mean(mean(PSNR30));
snr30.PSNR30_av_av_diff = mean(mean(PSNR30_diff));
snr30.SSIM30 = SSIM30;
snr30.SSIM30_diff = SSIM30_diff;
snr30.SSIM30_av = mean(SSIM30,2);
snr30.SSIM30_diff_av = mean(SSIM30_diff,2);
snr30.SSIM30_av_over_images = mean(SSIM30);
snr30.SSIM30_diff_av_over_images = mean(SSIM30_diff);
snr30.SSIM30_av_av = mean(mean(SSIM30));
snr30.SSIM30_av_av_diff = mean(mean(SSIM30_diff));
snr30.IOuts = IOuts;
snr30.Iins = noisy_Ims;

save([outpath '\paramsin.mat'], 'paramsin');
save([outpath '\paramsout.mat'], 'paramsout');
save([outpath '\snr30.mat'], 'snr30');
save([outpath '\rmse30.mat'], 'rmse30');
copyfile('tldecon_mf_result.m',outpath,'f');
copyfile('tldecon_mf1.m',outpath,'f');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% close all;
% clear all;
