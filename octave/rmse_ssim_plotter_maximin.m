p = 6; % # of sources
n = 1; % # of noise realizations
rmse30_av=zeros(sizeNU,sizeS);
rmse30_av_diff=zeros(sizeNU,sizeS);
ssim30_av=zeros(sizeNU,sizeS);
ssim30_av_diff=zeros(sizeNU,sizeS);

for j=1:n
     rmse30_av=rmse30_av+squeeze(max(rmse30(:,:,:,j),[],1));
     ssim30_av=ssim30_av+squeeze(min(SSIM30(:,:,:,j),[],1));
     rmse30_av_diff=rmse30_av_diff+squeeze(max(rmse30_diff(:,:,:,j),[],1));
     ssim30_av_diff=ssim30_av_diff+squeeze(min(SSIM30_diff(:,:,:,j),[],1));
end

rmse30_av=rmse30_av/(n);
ssim30_av=ssim30_av/(n);
rmse30_av_diff=rmse30_av_diff/(n);
ssim30_av_diff=ssim30_av_diff/(n);

figure, contour(s_array,nu_array,rmse30_av,100)
title('(PSNR)Opt Params k3 p6 objs128 Noiseless \nu-vs-s');
xlabel('s');
ylabel('\nu');

figure, contour(s_array,nu_array,ssim30_av,100)
title('(SSIM)Opt Params k3 p6 objs128 Noiseless \nu-vs-s');
xlabel('s');
ylabel('\nu');

figure, contour(s_array,nu_array,rmse30_av_diff,100)
title('(PSNR\_diff)Opt Params k3 p6 objs128 Noiseless \nu-vs-s');
xlabel('s');
ylabel('\nu');

figure, contour(s_array,nu_array,ssim30_av_diff,100)
title('(SSIM\_diff)Opt Params k3 p6 objs128 Noiseless \nu-vs-s');
xlabel('s');
ylabel('\nu');

% figure, plot(nu_array,ssim30_av_diff);
% title('(SSIM\_diff)Opt Params k3 p6 objs128 Noiseless \nu at s=0.01')
% xlabel('\nu')
% ylabel('SSIM_diff')
