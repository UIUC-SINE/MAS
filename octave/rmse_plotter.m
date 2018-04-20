p = 6; % # of sources
n = 1; % # of noise realizations
rmse30_av=zeros(sizeNU,sizeS);
ssim30_av=zeros(sizeNU,sizeS);
for i=1:p
    for j=1:n
        rmse30_av=rmse30_av+squeeze(rmse30(i,:,:,j));
        ssim30_av=ssim30_av+squeeze(SSIM30(i,:,:,j));
    end
end
rmse30_av=rmse30_av/(p*n);
ssim30_av=ssim30_av/(p*n);

figure, contour(s_array,nu_array,rmse30_av,100)
title('(PSNR)Opt Params k3 p6 objs128 30SNR \nu-vs-s');
xlabel('s');
ylabel('\nu');

figure, contour(s_array,nu_array,ssim30_av,100)
title('(SSIM)Opt Params k3 p6 objs128 30SNR \nu-vs-s');
xlabel('s');
ylabel('\nu');

% figure, plot(nu_array,rmse30_av);
% title('Opt Params k4 p5 pine145 30SNR \nu at s=0.01')
% xlabel('\nu')
% ylabel('RMSE')
