% measurement_simulation

% psf_initializer;
image_initializer;

% convLength='same';
%I: orignal images
%Im: noise free measurements
%noisy_Im: noisy measurements

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PSF GENERATION
incoherentPsf = zeros(sizeHy,sizeHx,s,s);
diffLimitingPsf = zeros(sizeHy,sizeHx,s);
H = zeros(k*sizeHy,s*sizeHx);

for i = 1:k
    for j = 1:s
        if i==j
            % Incoherent psf for the jth source at distance d(i)
            [incoherentPsf(:,:,i,j), diffLimitingPsf(:,:,i),di(i,j)]=incoherentPsf5(lambda(j),D,w,defocusAmount(i,j),sizeHx,pixelsize,diffLimitedCutOffFreq);
            if i==1
                factor=sum(sum(incoherentPsf(:,:,i,j)));
            end
            incoherentPsf(:,:,i,j)=incoherentPsf(:,:,i,j)/factor;
        else
            % Incoherent psf for the jth source at distance d(i)
            [incoherentPsf(:,:,i,j), ~,di(i,j)]=incoherentPsf5(lambda(j),D,w,defocusAmount(i,j),sizeHx,pixelsize,diffLimitedCutOffFreq);
            incoherentPsf(:,:,i,j)=incoherentPsf(:,:,i,j)/factor;
        end
        % All psfs in a single matrix
        H((i-1)*sizeHy+1:i*sizeHy,(j-1)*sizeHx+1:j*sizeHx) = incoherentPsf(:,:,i,j);
    end
end

%diff limiting psf
for i = 1:s
    % Diffraction limiting psf of i th source
    [~, diffLimitingPsf(:,:,i),~]=incoherentPsf5(lambda(i),D,w,0,sizeHx,pixelsize,diffLimitedCutOffFreq);
end

%diff limiting psf end
H_old = H;
% maske = randn(size(H))>0.5;
% H = ifft2(fft2(H .* maske));
% H = H .* maske;
% mutual_coherence(H)

%% For comparison, compute the diffraction-limited input images: this is the best possible reconstruction we could hope for
%% without using additional prior information

% Compute convolution of the input images with the diffraction-limiting psfs
for i = 1:s
    I_diff((i-1)*aa+1:i*aa,:) = conv2(indexer(I,i,1,aa),diffLimitingPsf(:,:,i),'same');
end

%% Generate blurred (+ noisy) measurements using psfs and ground truth images
psfs = H;
psfsize = sizeHx;
f_psfs = block_fft2(psfs, psfsize, aa); %taking fft2 of psfs
f_I = block_fft2(I, aa, aa); %taking fft2 of images

f_Im = my_mul(f_psfs, f_I, aa); %calculating noise free measurements in f domain

Im = block_ifft2(f_Im, aa); %noise free measurements in time domain

if add_noise == 0
    noisy_Im = Im;
else
    noisy_Im = block_imposeNoise(Im, aa, snrn); %noisy measurements in time domain
end

% Because measurements can not be below zero, set the noisy nonnegative
% values to zero
noisy_Im = max(0,noisy_Im);
 
%% Coded measurements
% Code = ones(aa); 
load('code_mtx_uncolored');
load('code_mtx_colored');
Code_mtx = [];

for i = 1 : p    
    Code = rand(aa)>0.5;
    Code_mtx = [Code_mtx; Code];
end
if color == 1
    Code_mtx = code_mtx_colored;
else
    Code_mtx = code_mtx_uncolored;
end
    

f_I_C = block_fft2(Code_mtx.*I, aa, aa); %taking fft2 of images

f_Im_C = my_mul(f_psfs, f_I_C, aa); %calculating noise free measurements in f domain

Im_C = block_ifft2(f_Im_C, aa); %noise free measurements in time domain

if add_noise == 0
    noisy_Im_C = Im_C;
else
    noisy_Im_C = block_imposeNoise(Im_C, aa, snrn); %noisy measurements in time domain
end

% noisy_Im_C = Im_C + noise;

% Because measurements can not be below zero, set the noisy nonnegative
% values to zero
noisy_Im_C = max(0,noisy_Im_C);