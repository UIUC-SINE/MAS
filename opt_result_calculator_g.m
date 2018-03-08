% by Fatih Çaðatay Akyön & Ulaþ Kamacý 2017
% fatihcagatayakyon@gmail.com & kamaci.ulas@gmail.com
%% Average PSNR Calculator for Optimum Parameters

%Multiple Image function version of opt_result_calculator_d.m script.
%Read descriptions on the file "opt_result_calculator_d.m".

% INPUTS
%image related:
aa=128; %image size
factor=1/4; %image resize factor
snrn = 25; %SNR of the measurements
nn = 5; %number of different noise realizations to be used

% output folder path:
outpath = '.\Results\3IMAGE3MEAS\25SNR_3IMAGE3MEAS_Opt_Param_5_MC_Results_2';
mkdir(outpath);

% optimum parameter set for 25 SNR:
nu_array = [40]; %opt. nu comes here
s_array = [0.2]; %opt. s comes here
lambda_array = [1];
imout_array25 = zeros(384,128,nn);
imin_array25 = zeros(384,128,nn);

psf_initializer;
% other algorithm related parameters:
paramsin.num = 50;
paramsin.numiter = 1; 
paramsin.n = 36;
paramsin.N = 128*128;
paramsin.C = 1e5;
paramsin.r = 1;
paramsin.psfsize = psfsize;
% paramsin.W01 = TV_mtx(36);
% paramsin.W02 = TV_mtx(36);
%paramsin.W01 = kron(dctmtx(6),dctmtx(6));
% paramsin.W01 = eye(36);
% paramsin.W02 = eye(36);
paramsin.ct = 1;
paramsin.nl = 1; 
paramsin.cini = 1;
paramsin.co = 0;
load('learned_transform3.mat');
load('diff_limited_images.mat');
paramsin.W01=learned_transform3;
paramsin.num_first = 5; %IS iteration number

% PARAMETER SCANNER

sizeNU = size(nu_array,2);
sizeS = size(s_array,2);
sizeLAMBDA = size(lambda_array,2);

counter=1;
rmse25=NaN(3,sizeNU,sizeS,sizeLAMBDA,nn);
PSNR25=NaN(3,nn);
PSNR25_diff=NaN(3,nn);
loading = 'Loading %';

for n=1:nn
    iternum = 1;
    measurement_simulation_g;
    for k=1:sizeNU
        paramsin.nu = nu_array(k)/(128*128);
        for l=1:sizeS
            paramsin.s = s_array(l)*ones(1,1000);
            for m=1:sizeLAMBDA
                paramsin.lambda0 = lambda_array(m);
                close(figure(6));
                [IOut,paramsout]= tldecon_g2(I,noisy_Im,psfs,paramsin);
                IOut1=indexer(IOut,1,1,aa); IOut2=indexer(IOut,2,1,aa); IOut3=indexer(IOut,3,1,aa);
                
                imin_array25(:,:,n)=noisy_Im;
                imout_array25(:,:,n)=IOut;
                rmse25(1,k,l,m,n) = 1/128*norm(I1_diff-IOut1,'fro');
                rmse25(2,k,l,m,n) = 1/128*norm(I2_diff-IOut2,'fro');
                rmse25(3,k,l,m,n) = 1/128*norm(I3_diff-IOut3,'fro');
                PSNR25(1,n) = paramsout.PSNR01;
                PSNR25(2,n) = paramsout.PSNR02;
                PSNR25(3,n) = paramsout.PSNR03;
                PSNR25_diff(1,n) = paramsout.PSNR01_diff;
                PSNR25_diff(2,n) = paramsout.PSNR02_diff;
                PSNR25_diff(3,n) = paramsout.PSNR03_diff;
                disp(loading); disp([counter*100/(nn*sizeNU*sizeS*sizeLAMBDA) n k l m rmse25(1,k,l,m,n) rmse25(2,k,l,m,n)]);
                
                close(figure(6));
                figure(6), plot(paramsout.PSNR1_diff);
                hold on
                plot(paramsout.PSNR2_diff)
                hold on
                plot(paramsout.PSNR3_diff)
                %title(['NR=',num2str(n),'\nu=',num2str(paramsin.nu*128^2),' s=',num2str(paramsin.s(1)),'\lambda=',num2str(paramsin.lambda0),'PSNR1=',num2str(paramsout.PSNR01),'PSNR2=',num2str(paramsout.PSNR02)]); grid minor;
                title({['NR=',num2str(n),',',' \nu=',num2str(paramsin.nu*128^2),',',' s=',num2str(paramsin.s(1)),',',' \lambda=',num2str(paramsin.lambda0)],['PSNR1=',num2str(paramsout.PSNR01_diff)],[' PSNR2=',num2str(paramsout.PSNR02_diff)],[' PSNR3=',num2str(paramsout.PSNR03_diff)]}); grid minor;
                
                saveas(figure(6),[outpath, '\','NR=',num2str(n),'_',num2str(iternum)],'png');
                saveas(figure(6),[outpath, '\','NR=',num2str(n),'_',num2str(iternum)],'fig');
                iternum = iternum + 1;
                counter = counter+1;
            end
        end
    end
end


snr25.PSNR25=PSNR25;
snr25.PSNR25_diff=PSNR25_diff;
snr25.PSNR25_av1 = sum(squeeze(PSNR25(1,:)))/nn;
snr25.PSNR25_av2 = sum(squeeze(PSNR25(2,:)))/nn;
snr25.PSNR25_av3 = sum(squeeze(PSNR25(3,:)))/nn;
snr25.PSNR25_av1_diff = sum(squeeze(PSNR25_diff(1,:)))/nn;
snr25.PSNR25_av2_diff = sum(squeeze(PSNR25_diff(2,:)))/nn;
snr25.PSNR25_av3_diff = sum(squeeze(PSNR25_diff(3,:)))/nn;
snr25.PSNR25_av = (snr25.PSNR25_av1 + snr25.PSNR25_av2 + snr25.PSNR25_av3)/3;
snr25.PSNR25_av_diff = (snr25.PSNR25_av1_diff + snr25.PSNR25_av2_diff + snr25.PSNR25_av3_diff)/3;
snr25.imout_array25 = imout_array25;
snr25.imin_array25 = imin_array25;

save([outpath '\paramsin.mat'], 'paramsin');
save([outpath '\paramsout.mat'], 'paramsout');
save([outpath '\rmse25.mat'], 'rmse25');
save([outpath '\snr25.mat'], 'snr25');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
close all;
clear all;

% INPUTS
%image related:
aa=128; %image size
factor=1/4; %image resize factor
snrn = 20; %SNR of the measurements
nn = 5; %number of different noise realizations to be used

% output folder path:
outpath = '.\Results\3IMAGE3MEAS\20SNR_3IMAGE3MEAS_Opt_Param_5_MC_Results';
mkdir(outpath);

% optimum parameter set for 20 SNR:
nu_array = [10]; %opt. nu comes here
s_array = [0.16]; %opt. s comes here
lambda_array = [1];
imout_array20 = zeros(384,128,nn);
imin_array20 = zeros(384,128,nn);

psf_initializer;
% other algorithm related parameters:
paramsin.num = 50;
paramsin.numiter = 1; 
paramsin.n = 36;
paramsin.N = 128*128;
paramsin.C = 1e5;
paramsin.r = 1;
paramsin.psfsize = psfsize;
% paramsin.W01 = TV_mtx(36);
% paramsin.W02 = TV_mtx(36);
%paramsin.W01 = kron(dctmtx(6),dctmtx(6));
% paramsin.W01 = eye(36);
% paramsin.W02 = eye(36);
paramsin.ct = 1;
paramsin.nl = 1; 
paramsin.cini = 1;
paramsin.co = 0;
load('learned_transform3.mat');
load('diff_limited_images.mat');
paramsin.W01=learned_transform3;
paramsin.num_first = 5; %IS iteration number

% PARAMETER SCANNER

sizeNU = size(nu_array,2);
sizeS = size(s_array,2);
sizeLAMBDA = size(lambda_array,2);

counter=1;
rmse20=NaN(3,sizeNU,sizeS,sizeLAMBDA,nn);
PSNR20=NaN(3,nn);
PSNR20_diff=NaN(2,nn);
loading = 'Loading %';

for n=1:nn
    iternum = 1;
    measurement_simulation_g;
    for k=1:sizeNU
        paramsin.nu = nu_array(k)/(128*128);
        for l=1:sizeS
            paramsin.s = s_array(l)*ones(1,1000);
            for m=1:sizeLAMBDA
                paramsin.lambda0 = lambda_array(m);
                close(figure(6));
                [IOut,paramsout]= tldecon_g2(I,noisy_Im,psfs,paramsin);
                IOut1=indexer(IOut,1,1,aa); IOut2=indexer(IOut,2,1,aa); IOut3=indexer(IOut,3,1,aa);
                
                imin_array20(:,:,n)=noisy_Im;
                imout_array20(:,:,n)=IOut;
                rmse20(1,k,l,m,n) = 1/128*norm(I1_diff-IOut1,'fro');
                rmse20(2,k,l,m,n) = 1/128*norm(I2_diff-IOut2,'fro');
                rmse20(3,k,l,m,n) = 1/128*norm(I3_diff-IOut3,'fro');
                PSNR20(1,n) = paramsout.PSNR01;
                PSNR20(2,n) = paramsout.PSNR02;
                PSNR20(3,n) = paramsout.PSNR03;
                PSNR20_diff(1,n) = paramsout.PSNR01_diff;
                PSNR20_diff(2,n) = paramsout.PSNR02_diff;
                PSNR20_diff(3,n) = paramsout.PSNR03_diff;
                disp(loading); disp([counter*100/(nn*sizeNU*sizeS*sizeLAMBDA) n k l m rmse20(1,k,l,m,n) rmse20(2,k,l,m,n)]);
                
                close(figure(6));
                figure(6), plot(paramsout.PSNR1_diff);
                hold on
                plot(paramsout.PSNR2_diff)
                hold on
                plot(paramsout.PSNR3_diff)
                %title(['NR=',num2str(n),'\nu=',num2str(paramsin.nu*128^2),' s=',num2str(paramsin.s(1)),'\lambda=',num2str(paramsin.lambda0),'PSNR1=',num2str(paramsout.PSNR01),'PSNR2=',num2str(paramsout.PSNR02)]); grid minor;
                title({['NR=',num2str(n),',',' \nu=',num2str(paramsin.nu*128^2),',',' s=',num2str(paramsin.s(1)),',',' \lambda=',num2str(paramsin.lambda0)],['PSNR1=',num2str(paramsout.PSNR01_diff)],[' PSNR2=',num2str(paramsout.PSNR02_diff)],[' PSNR3=',num2str(paramsout.PSNR03_diff)]}); grid minor;
                
                saveas(figure(6),[outpath, '\','NR=',num2str(n),'_',num2str(iternum)],'png');
                saveas(figure(6),[outpath, '\','NR=',num2str(n),'_',num2str(iternum)],'fig');
                iternum = iternum + 1;
                counter = counter+1;
            end
        end
    end
end

snr20.PSNR20=PSNR20;
snr20.PSNR20_diff=PSNR20_diff;
snr20.PSNR20_av1 = sum(squeeze(PSNR20(1,:)))/nn;
snr20.PSNR20_av2 = sum(squeeze(PSNR20(2,:)))/nn;
snr20.PSNR20_av3 = sum(squeeze(PSNR20(3,:)))/nn;
snr20.PSNR20_av1_diff = sum(squeeze(PSNR20_diff(1,:)))/nn;
snr20.PSNR20_av2_diff = sum(squeeze(PSNR20_diff(2,:)))/nn;
snr20.PSNR20_av3_diff = sum(squeeze(PSNR20_diff(3,:)))/nn;
snr20.PSNR20_av = (snr20.PSNR20_av1 + snr20.PSNR20_av2 + snr20.PSNR20_av3)/3;
snr20.PSNR20_av_diff = (snr20.PSNR20_av1_diff + snr20.PSNR20_av2_diff + snr20.PSNR20_av3_diff)/3;
snr20.imout_array20 = imout_array20;
snr20.imin_array20 = imin_array20;

save([outpath '\paramsin.mat'], 'paramsin');
save([outpath '\paramsout.mat'], 'paramsout');
save([outpath '\rmse20.mat'], 'rmse20');
save([outpath '\snr20.mat'], 'snr20');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
close all;
clear all;

% INPUTS
%image related:
aa=128; %image size
factor=1/4; %image resize factor
snrn = 30; %SNR of the measurements
nn = 5; %number of different noise realizations to be used

% output folder path:
outpath = '.\Results\3IMAGE3MEAS\30SNR_3IMAGE3MEAS_Opt_Param_5_MC_Results_(new parameters)';
mkdir(outpath);

% optimum parameter set for 30 SNR:
nu_array = [90]; %opt. nu comes here
s_array = [0.2]; %opt. s comes here
lambda_array = [1];
imout_array30 = zeros(384,128,nn);
imin_array30 = zeros(384,128,nn);

psf_initializer;
% other algorithm related parameters:
paramsin.num = 50;
paramsin.numiter = 1; 
paramsin.n = 36;
paramsin.N = 128*128;
paramsin.C = 1e5;
paramsin.r = 1;
paramsin.psfsize = psfsize;
% paramsin.W01 = TV_mtx(36);
% paramsin.W02 = TV_mtx(36);
%paramsin.W01 = kron(dctmtx(6),dctmtx(6));
% paramsin.W01 = eye(36);
% paramsin.W02 = eye(36);
paramsin.ct = 1;
paramsin.nl = 1; 
paramsin.cini = 1;
paramsin.co = 0;
load('learned_transform3.mat');
load('diff_limited_images.mat');
paramsin.W01=learned_transform3;
paramsin.num_first = 5; %IS iteration number

% PARAMETER SCANNER

sizeNU = size(nu_array,2);
sizeS = size(s_array,2);
sizeLAMBDA = size(lambda_array,2);
counter=1;

rmse30=NaN(3,sizeNU,sizeS,sizeLAMBDA,nn);
PSNR30=NaN(3,nn);
PSNR30_diff=NaN(2,nn);
loading = 'Loading %';

for n=1:nn
    iternum = 1;
    measurement_simulation_g;
    for k=1:sizeNU
        paramsin.nu = nu_array(k)/(128*128);
        for l=1:sizeS
            paramsin.s = s_array(l)*ones(1,1000);
            for m=1:sizeLAMBDA
                paramsin.lambda0 = lambda_array(m);
                close(figure(6));
                [IOut,paramsout]= tldecon_g2(I,noisy_Im,psfs,paramsin);
                IOut1=indexer(IOut,1,1,aa); IOut2=indexer(IOut,2,1,aa); IOut3=indexer(IOut,3,1,aa);
                
                imin_array30(:,:,n)=noisy_Im;
                imout_array30(:,:,n)=IOut;
                rmse30(1,k,l,m,n) = 1/128*norm(I1_diff-IOut1,'fro');
                rmse30(2,k,l,m,n) = 1/128*norm(I2_diff-IOut2,'fro');
                rmse30(3,k,l,m,n) = 1/128*norm(I3_diff-IOut3,'fro');
                PSNR30(1,n) = paramsout.PSNR01;
                PSNR30(2,n) = paramsout.PSNR02;
                PSNR30(3,n) = paramsout.PSNR03;
                PSNR30_diff(1,n) = paramsout.PSNR01_diff;
                PSNR30_diff(2,n) = paramsout.PSNR02_diff;
                PSNR30_diff(3,n) = paramsout.PSNR03_diff;
                disp(loading); disp([counter*100/(nn*sizeNU*sizeS*sizeLAMBDA) n k l m rmse30(1,k,l,m,n) rmse30(2,k,l,m,n)]);
                
                close(figure(6));
                figure(6), plot(paramsout.PSNR1_diff);
                hold on
                plot(paramsout.PSNR2_diff)
                hold on
                plot(paramsout.PSNR3_diff)
                title({['NR=',num2str(n),',',' \nu=',num2str(paramsin.nu*128^2),',',' s=',num2str(paramsin.s(1)),',',' \lambda=',num2str(paramsin.lambda0)],['PSNR1=',num2str(paramsout.PSNR01_diff)],[' PSNR2=',num2str(paramsout.PSNR02_diff)],[' PSNR3=',num2str(paramsout.PSNR03_diff)]}); grid minor;
                [IOut,paramsout]= tldecon_g2(I,noisy_Im,psfs,paramsin);
                
                saveas(figure(6),[outpath, '\','NR=',num2str(n),'_',num2str(iternum)],'png');
                saveas(figure(6),[outpath, '\','NR=',num2str(n),'_',num2str(iternum)],'fig');
                iternum = iternum + 1;
                counter = counter + 1;
            end
        end
    end
end

snr30.PSNR30=PSNR30;
snr30.PSNR30_diff=PSNR30_diff;
snr30.PSNR30_av1 = sum(squeeze(PSNR30(1,:)))/nn;
snr30.PSNR30_av2 = sum(squeeze(PSNR30(2,:)))/nn;
snr30.PSNR30_av3 = sum(squeeze(PSNR30(3,:)))/nn;
snr30.PSNR30_av1_diff = sum(squeeze(PSNR30_diff(1,:)))/nn;
snr30.PSNR30_av2_diff = sum(squeeze(PSNR30_diff(2,:)))/nn;
snr30.PSNR30_av3_diff = sum(squeeze(PSNR30_diff(3,:)))/nn;
snr30.PSNR30_av = (snr30.PSNR30_av1 + snr30.PSNR30_av2 + snr30.PSNR30_av3)/3;
snr30.PSNR30_av_diff = (snr30.PSNR30_av1_diff + snr30.PSNR30_av2_diff + snr30.PSNR30_av3_diff)/3;
snr30.imout_array30 = imout_array30;
snr30.imin_array30 = imin_array30;

save([outpath '\paramsin.mat'], 'paramsin');
save([outpath '\paramsout.mat'], 'paramsout');
save([outpath '\rmse30.mat'], 'rmse30');
save([outpath '\snr30.mat'], 'snr30');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%