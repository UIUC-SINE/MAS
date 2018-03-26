%% Photon Sieve PSF Script

%%%%%%%% Parameters %%%%%%%%
sizeHx = 51;  % PSFsize , MUST be of ODD length!!!
sizeHy = sizeHx;

% specify the measurement planes
dn = linspace(0,1,1000); % fill this with normalized values
k = length(dn); % number of measurement planes

% Wavelengths of the monochromatic sources
lambda(1)=33.4*10^(-9);      %1nd source wavelength
lambda(2)=33.5*10^(-9);      %2rd source wavelength

s = length(lambda); % num of sources

% Photon sieve parameters (VARY D to change the defocusing amount)
D=25*10^(-3);        %diameter of the outer zone
w= 5*10^(-6);        %width of the smallest zone

%% Other photon sieve parameters following from above choices
f = zeros(s,1);
DOF = zeros(s,1);

for i = 1:s
    f(i) = D*w/lambda(i);       %first order focal length
    DOF(i) = 2*w^2/lambda(i);   %depth of focus
end

%% Resolution
% For the coherent case, largest diffraction-limited bandwidth allowed at the image plane (determined by the aperture size)
% di=min(f_1,f_2);  %largest diff. limited bandwidth occurs at the smallest distance to the photon sieve
% diffLimitedBandwidth_1=D/(lambda_1*di);
% diffLimitedBandwidth_2=D/(lambda_2*di);
diffLimitedCutOffFreq=1/w;   %same as D/(lambda_1*f_1), D/(lambda_2*f_2)

% Rayleigh resolution given by 1.22 \Delta, where \Delta is the width of the smallest zone
RayleighResolution=1.22*w;     %same as 1.22/diffLimitedBandwidth_1
SparrowResolution=0.94*w;
AbbeResolution=w;
maxPossibleResolution=1/(2*diffLimitedCutOffFreq); %correspond to the cut-off frequency due to diffraction

% Choose a unique sampling interval in the space domain based on the maxBandwidth (for FFT-based psf computation)
pixelsize=maxPossibleResolution;  %Nyquist sampling interval=maxPossibleResolution

%% Determine the measurement planes and
%% find the corresponding defocusing amounts for each source

% Focal Planes:      fs  fs-1        f1
% Illustrantion:     |---|---|- ... -|
% Normalization:     0               1

% construct dn with increasing order

% dn = [0 0.25 0.50 0.75 1];
% dn = [0.1493    0.2575    0.6507    0.9543]; % fill this with normalized values
% dn = [0.1493    0.5325    0.8412]; % fill this with normalized values

% dn = [0.20 0.43 0.60 0.81];
% dn = [0 0.25 0.50 0.75];

width = f(1) - f(s);

d = width * dn + f(s); % elements of d are true d(i) values
d = fliplr(d);


defocusAmount = zeros(s,s);

% Defocusing amount for the jth source when it is measured at the ith focal plane
for i = 1:k
    for j = 1:s
        defocusAmount(i,j) = (d(i)-f(j))/DOF(j); % defocusAmount x DOF away from its focus point
    end
end

%% PSF GENERATION
incoherentPsf = zeros(sizeHy,sizeHx,s,s);
diffLimitingPsf = zeros(sizeHy,sizeHx,s);
H = zeros(k*sizeHy,s*sizeHx);

printf('starting iterations')
for i = 1:k
    if mod(i, 10) == 0
        printf('Iterations: %d\n', i)
        fflush(stdout)
    endif
    
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

%% Generate blurred (+ noisy) measurements using psfs and ground truth images
psfs = H;
