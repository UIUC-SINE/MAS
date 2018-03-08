function [incoherentPsf, diffLimitingPsf,di]=incoherentPsf5(lambda,D,w,defocusAmount,M,deltat,maxBandwidth)
% Compute the approximate psf in the space domain for the incoherent case and given
%   - lambda:   source wavelength (in meters)
%   - D:        diameter of the outer zone (in meters)
%   - w:        width of the smallest zone (in meters)
%   - defocusAmount:    Distance from the focal plane in terms of DOF (defocusAmount x DOF away from the focus point)
%   - M:        total number of samples (MUST BE ODD!)
%   - deltat:   sampling interval in the space domain (for FFT computation)

% Difference with the incoherentPsf4.m: M, total number of samples, is ODD
% Difference with the incoherentPsf3.m: the input term K is removed
% Difference with the incoherentPsf.m: sampling interval is input to the method


%% Parameters following from the given parameters

% Photon sieve
f1=D*w/lambda;          %first order focal length
DOF=2*w^2/lambda;       %depth of focus



% %OFK Distance to image plane
% di=f1+defocusAmount * DOF - 0.1;  %distance to the image plane

% Distance to image plane
di=f1+defocusAmount * DOF;  %distance to the image plane

% Diffraction-limited bandwidth observed at the image plane (determined by the aperture size)
diffLimitedBandwidth=D/(lambda*di); %the factor 2 is due to the incoherence

% Given the pixel size and number of pixels, compute the
% sampling interval in the frequency domain
deltaf=1/(M*deltat); %assuming M and deltat are the same for both dimensions x and y

%% First compute the COHERENT psf in the frequency domain 

% Defocusing parameter in the Fresnel formula for the first order, i.e. m=1, and when d_s>>d_i and d_s>>f_m
epsilon1= - defocusAmount*DOF/(f1*(f1+defocusAmount*DOF));

% Evaluate the scaled aperture function at the sampled frequency points at
% points -M/2,...,0,1,...,M/2-1
m=-(M-1)/2:(M-1)/2;
evalPoints=deltaf*m;

% Optical transfer func: circ(f_x/diffLimitedBandwidth, f_y/diffLimitedBandwidth)
[row,col] = meshgrid(evalPoints, evalPoints);
OPTfocused = double(sqrt(row.^2+col.^2) <= diffLimitedBandwidth/2); %at the focus

% if epsilon1==0 %at the focus
%     
%     coherentOPT = OPTfocused;
% else           %away from focus
    coherentOPT = OPTfocused .* exp(1i*pi*epsilon1*lambda*di^2*(row.^2+col.^2));
% end

% WARNING: Above, multiplication factor (lambda*di)^2 is neglected

%% Compute the COHERENT psf in the space domain using FFTs

% Coherent psf in the space domain
coherentPsf=fftshift(ifft2(ifftshift(coherentOPT))); 
%coherentPsf=coherentPsf*prod(size(coherentOPT));

%Above, 
% 1) call ifftshift first to to move the DC term from the centre of the 2D array to the top-left corner
% 2) after ifft, call fftshift since the signal index=-N/2,...,N/2-1

% Incoherent psf in the space domain: magnitude square of the coherent psf
incoherentPsf=abs(coherentPsf).^2;

%% Also compute the diffraction-limited OPT for the input image, 
%  i.e. circ((lambda*d_i/2D) f_x, (lambda*d_i/2D) f_y)

% In the frequency domain
[row,col] = meshgrid(evalPoints, evalPoints);
diffLimitingOPT = double(sqrt(row.^2+col.^2) <= maxBandwidth); %at the focus

% In the space domain
diffLimitingPsf=fftshift(ifft2(ifftshift(diffLimitingOPT))); 

%% PLOTS: For verification, display the psfs in the space and frequency domains

% % Compute the INCOHERENT psf in the frequency domain
% incoherentOPT=fftshift(fft2(ifftshift(incoherentPsf)));

% % Display the coherent OPT in the frequency domain
% figure; imshow(abs(coherentOPT),[min(abs(coherentOPT(:))) max(abs(coherentOPT(:)))]); title('Magnitude of coherent OPT')
% figure; imshow(real(coherentOPT),[min(real(coherentOPT(:))) max(real(coherentOPT(:)))]); title('Real part of coherent OPT')
% figure; imshow(imag(coherentOPT),[min(imag(coherentOPT(:))) max(imag(coherentOPT(:)))]); title('Imaginary part of coherent OPT')

% Display the incoherent psf
%figure; imagesc(incoherentPsf,[min(incoherentPsf(:)) max(incoherentPsf(:))]); colormap(gray); axis image;  title('Incoherent psf') %2D magnitude response
%figure; imshow(incoherentPsf,[min(incoherentPsf(:)) max(incoherentPsf(:))]); title('Incoherent psf') %2D magnitude response
%figure; imshow(real(coherentPsf),[min(real(coherentPsf(:))) max(real(coherentPsf(:)))]); title('Real part of coherent psf') 
%figure; imshow(imag(coherentPsf),[min(imag(coherentPsf(:))) max(imag(coherentPsf(:)))]); title('Imaginary part of coherent psf') 

%figure; plot(incoherentPsf((M-1)/2,:));   %1D incoherent psf at the center

%figure; plot(real(coherentPsf(M/2+1,:)));  %1D real part of the coherent psf at the center
%figure; plot(imag(coherentPsf(M/2+1,:)));  %1D imaginary part of the coherent psf at the center

% % Display the incoherent OPT in the frequency domain
% figure; imshow(abs(incoherentOPT),[min(abs(incoherentOPT(:))) max(abs(incoherentOPT(:)))]); title('Magnitude of incoherent OPT')
% figure; imshow(real(incoherentOPT),[min(real(incoherentOPT(:))) max(real(incoherentOPT(:)))]); title('Real part of incoherent OPT')
% figure; imshow(imag(incoherentOPT),[min(imag(incoherentOPT(:))) max(imag(incoherentOPT(:)))]); title('Imaginary part of incoherent OPT')

%[min(imag(incoherentOPT(:))) max(imag(incoherentOPT(:)))]

% % Display in 3D view
%figure; surf(row,col,abs(incoherentOPT)) %to check chinese hat function



