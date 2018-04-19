close(figure(13));
figure(13), plot(paramsout.Werror);
title('W-{error}, NOISY-MEAS-DCT-W/-20INNER-IUs-1INNER-TUs');grid minor;

close(figure(14));
figure(14), plot(paramsout.sparsity1);
hold on
plot(paramsout.sparsity2);
legend('Sparsity 1','Sparsity 2');
title('Sparsity, NOISY-MEAS-DCT-W/-20INNER-IUs-1INNER-TUs');grid minor;

close(figure(15));
figure(15), plot(paramsout.sp1);
hold on
plot(paramsout.sp2);
legend('Sparsification Error 1','Sparsification Error 2');
title('Sparsification Error, NOISY-MEAS-DCT-W/-20INNER-IUs-1INNER-TUs');grid minor;

outpath = '.\Results\NOISY-MEAS-DCT-W/-20INNER-IUs-1INNER-TUs';
mkdir(outpath);

saveas(figure(6),[outpath '\PSNR-Out'],'png');
saveas(figure(9),[outpath '\Transform-Out'],'png');
saveas(figure(13),[outpath '\Werror'],'png');
saveas(figure(14),[outpath '\Sparsity'],'png');
saveas(figure(15),[outpath '\Sparsification Error'],'png');

saveas(figure(6),[outpath '\PSNR-Out'],'fig');
saveas(figure(9),[outpath '\Transform-Out'],'fig');
saveas(figure(13),[outpath '\Werror'],'fig');
saveas(figure(14),[outpath '\Sparsity'],'fig');
saveas(figure(15),[outpath '\Sparsification Error'],'fig');

save([outpath '\paramsin.mat'], 'paramsin');
save([outpath '\paramsout.mat'], 'paramsout');
save([outpath '\IOut1.mat'], 'IOut1');
save([outpath '\IOut2.mat'], 'IOut2');
%%
close(figure(13));
figure(13), plot(paramsout.Werror);
title('W-{error}, NOISY-MEAS-DCT-Initialized-TL-First-2');grid minor;

close(figure(14));
figure(14), plot(paramsout.sparsity1);
hold on
plot(paramsout.sparsity2);
legend('Sparsity 1','Sparsity 2');
title('Sparsity, NOISY-MEAS-DCT-Initialized-TL-First-2');grid minor;

close(figure(15));
figure(15), plot(paramsout.sp1);
hold on
plot(paramsout.sp2);
legend('Sparsification Error 1','Sparsification Error 2');
title('Sparsification Error, NOISY-MEAS-DCT-Initialized-TL-First-2');grid minor;

outpath = '.\Results\NOISY-MEAS-DCT-Initialized-TL-First-2';
mkdir(outpath);

saveas(figure(6),[outpath '\PSNR-Out'],'png');
saveas(figure(9),[outpath '\Transform-Out'],'png');
saveas(figure(13),[outpath '\Werror'],'png');
saveas(figure(14),[outpath '\Sparsity'],'png');
saveas(figure(15),[outpath '\Sparsification Error'],'png');

saveas(figure(6),[outpath '\PSNR-Out'],'fig');
saveas(figure(9),[outpath '\Transform-Out'],'fig');
saveas(figure(13),[outpath '\Werror'],'fig');
saveas(figure(14),[outpath '\Sparsity'],'fig');
saveas(figure(15),[outpath '\Sparsification Error'],'fig');

save([outpath '\paramsin.mat'], 'paramsin');
save([outpath '\paramsout.mat'], 'paramsout');
save([outpath '\IOut1.mat'], 'IOut1');
save([outpath '\IOut2.mat'], 'IOut2');
%%

close(figure(13));
figure(13), plot(paramsout.Werror);
title('W-{error} DCT-Initialized-s-0.16-lambda-0.1-10iter');grid minor;

close(figure(14));
figure(14), plot(paramsout.sparsity1);
hold on
plot(paramsout.sparsity2);
legend('Sparsity 1','Sparsity 2');
title('Sparsity, DCT-Initialized-s-0.16-lambda-0.1-10iter');grid minor;

close(figure(15));
figure(15), plot(paramsout.sp1);
hold on
plot(paramsout.sp2);
legend('Sparsification Error 1','Sparsification Error 2');
title('Sparsification Error, DCT-Initialized-s-0.16-lambda-0.1-10iter');grid minor;

outpath = '.\Results\RANDOM-IMAGES-FIXED-DCT-TL-First';
mkdir(outpath);

saveas(figure(6),[outpath '\PSNR-Out'],'png');
saveas(figure(9),[outpath '\Transform-Out'],'png');
saveas(figure(13),[outpath '\Werror'],'png');
saveas(figure(14),[outpath '\Sparsity'],'png');
saveas(figure(15),[outpath '\Sparsification Error'],'png');

saveas(figure(6),[outpath '\PSNR-Out'],'fig');
saveas(figure(9),[outpath '\Transform-Out'],'fig');
saveas(figure(13),[outpath '\Werror'],'fig');
saveas(figure(14),[outpath '\Sparsity'],'fig');
saveas(figure(15),[outpath '\Sparsification Error'],'fig');

save([outpath '\paramsin.mat'], 'paramsin');
save([outpath '\paramsout.mat'], 'paramsout');
save([outpath '\IOut1.mat'], 'IOut1');
save([outpath '\IOut2.mat'], 'IOut2');