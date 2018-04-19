for i=1:3
aa=128;
figure(15)
imshow(indexer(snr25.imin_array25(:,:,2),i,1,aa),[])
axis image
colormap gray
saveas(figure(15),['C:\Users\fatih\Dropbox\Ulas-Hilmi-Fatih\publications\conferences\GlobalSIP 2017\Presentation', '\NoisyMeas', num2str(i)],'png');
end

for i=1:3
aa=128;
figure(15)
imshow(indexer(snr25.imout_array25(:,:,2),i,1,aa),[])
axis image
colormap gray
saveas(figure(15),['C:\Users\fatih\Dropbox\Ulas-Hilmi-Fatih\publications\conferences\GlobalSIP 2017\Presentation', '\Recons', num2str(i)],'png');
end
