% image_initalizer

% This code generates the ground truth images, and set some parameters to
% be used in the PSF generation code. Note that each band of the ground
% truth images has minimum intensity of 0 and maximum intensity of 1.

if imsel == 0
    sizeHx = 51;  % PSFsize , MUST be of ODD length!!!
    sizeHy = sizeHx;
    I = zeros(p*aa,aa);
    for j=1:p
        II = imread(['SI1_',num2str(j),'.jpg']);
        II = double(rgb2gray(II));
        II = imresize(II,factor);
        II = II-min(II(:));
        II = II/max(max(II));
        I(1+(j-1)*aa:j*aa, 1:aa) = II;
    end
    clear II
    setPhotonSieveParameters3;
elseif imsel == 1
    sizeHx = 51;
    sizeHy = sizeHx;
    aa = 128;
    %     k = 4;
    %     p = 5;
    %     load('test_images_ofk.mat')
    %     I = a;
    %     load('pine_samples_5_adjacent.mat');
    %     I = pine_samples_5_adjacent;
    %     load('pines_5_selected_ofk.mat');
    load('reflectances_6_selected.mat')
    %     load('ozo_pine_selected_33.mat')
    %     I = pine_selected;
    m0 = 230;
    n0 = 630;
    for j=1:p
        temp = indexer(reflectances_selected,j,1,820);
        I(1+(j-1)*aa:j*aa, 1:aa) = temp(m0:m0+127,n0:n0+127);
    end
    I = I/max(I(:));
    clear temp;
    setPhotonSieveParameters3;
else
    error('set imsel = 0 or 1');
end
