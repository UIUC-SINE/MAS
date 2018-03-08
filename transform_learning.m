%% Transform Learning from Training Data
% Consists of Transform Update and Sparse Coding Steps
% Both 2D and 3D TL codes are included for comparison
% Outputs the learned transform (both 2D and 3D) from the given data cube

% Parameters and Initializations
aa = 128;
bb = 128;
p = 6;
load('reflectances_6_selected.mat')
%     load('ozo_pine_selected_33.mat')
%     I = pine_selected;
m0 = 230;
n0 = 630;
I = zeros(aa,aa,p);
for j=1:p
    temp = indexer(reflectances_selected,j,1,820);
    I(1:aa, 1:aa, j) = temp(m0:m0+127,n0:n0+127);
end
I = I/max(I(:));
for i=1:p
    I_(1+(i-1)*aa:i*aa,:) = I(:,:,i);
end
clear temp;
n1 = 81;
n2 = 6;
N2 = aa*aa;
N = aa*aa*p;
l0 = 0.1; % lambda parameter
sp = 0.040;
numiter3d = 5;
numiter2d = 5;
W2d = kron(dctmtx(sqrt(n1)),dctmtx(sqrt(n1)));
W3d = kron(dctmtx(n2),kron(dctmtx(sqrt(n1)),dctmtx(sqrt(n1))));
Werror3d = zeros(1,numiter3d);
Werror2d = zeros(1,numiter2d);

%%%%%%%%%%%%%%%% 3D TL %%%%%%%%%%%%%%%%
[rows,cols,slices] = ind2sub([sqrt(n1),sqrt(n1),n2],1:n1*n2);

%Creating all image patches (including wrap around patches)
TE = zeros(n1*n2,N2*p);
Ibs = cat(3,I,I(:,:,1:n2-1));
Ib1 = [Ibs Ibs(:,1:(sqrt(n1)-1),:);Ibs(1:(sqrt(n1)-1),:,:) Ibs(1:(sqrt(n1)-1),1:(sqrt(n1)-1),:)];
for j=1:n1*n2
    TE(j,:) = reshape(Ib1(rows(j):rows(j)+aa-1,cols(j):cols(j)+aa-1,slices(j):slices(j)+p-1),aa*aa*p,1);
end

X1 = W3d*TE;
[s1]=sort(abs(X1(:)),'descend');
X3d = X1.*(abs(X1) >= s1(round(sp*n1*n2*N2*p))); %initialize sparse codes (for the first TLDECON iteration)

%Transform learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Create a random permutation for the case when only a subset of patches are used in learning
if(N<N2*p)
    de=randperm(N2*p);
else
    de=1:N2*p;
end

%Transform learning iterations
for pt=1:numiter3d
    Witer3d = W3d;
    if(pt==1)
        YH = TE(:,de(1:N)); %Training data
        XH = X3d(:,de(1:N)); %Sparse codes
        l3 = l0*N; l2=l3; %Weights on the Frobenius norm and negative log-determinant terms in transform learning
        [U,S,V]=svd((YH*YH') + (0.5*l3*eye(n1*n2)));
        LL2=(inv(U*(S^(1/2))*V'));  %inverse square root factor used in transform learning
    end
    
    %Transform Update
    [GW,Si,R]=svd(LL2*(YH*XH'));
    sig=diag(Si);
    gamm=(1/2)*(sig + (sqrt((sig.^2) + 2*l2)));
    B=R*(diag(gamm))*GW';
    W3d=B*(LL2);
    
    Werror3d(pt)=norm(Witer3d - W3d,'fro')/norm(W3d,'fro');
    figure (3), plot(Werror3d); title('W3d-itererror');
    
    %Sparse Coding
    if(pt<numiter3d)
        XH=W3d*TE(:,de(1:N));
        [s1]=sort(abs(XH(:)),'descend');
        XH = XH.*(abs(XH) >= s1(round(sp*n1*n2*N)));
    end
end

%%%%%%%%% Sparse Coding and Image Compression with the Learned Transform %%%%%%%%%

X1 = W3d*TE;
[s1]=sort(abs(X1(:)),'descend');
X3d = X1.*(abs(X1) >= s1(round(sp*n1*n2*N2*p))); %find all the sparse codes with the updated transform
Z3d = W3d\X3d;

IMout1=zeros(size(Ib1));

for j=1:n1*n2
    IMout1(rows(j):rows(j)+aa-1,cols(j):cols(j)+aa-1,slices(j):slices(j)+p-1) = ...
        IMout1(rows(j):rows(j)+aa-1,cols(j):cols(j)+aa-1,slices(j):slices(j)+p-1) + ...
        reshape(Z3d(j,:),aa,aa,p);
end

IMout1(:,:,1:n2-1) = IMout1(:,:,1:n2-1) + IMout1(:,:,p+1:end);
Im_3d=IMout1(1:aa,1:bb,1:p);
Im_3d(1:(sqrt(n1)-1),:,:)= Im_3d(1:(sqrt(n1)-1),:,:)+ IMout1(aa+1:size(IMout1,1),1:bb,1:p);
Im_3d(:, 1:(sqrt(n1)-1),:)=Im_3d(:, 1:(sqrt(n1)-1),:) + IMout1(1:aa,bb+1:size(IMout1,2),1:p);
Im_3d(1:(sqrt(n1)-1),1:(sqrt(n1)-1),:)= Im_3d(1:(sqrt(n1)-1),1:(sqrt(n1)-1),:)+ IMout1(aa+1:size(IMout1,1),bb+1:size(IMout1,2),1:p);

Im_3d = Im_3d/(n1*n2); % normalize the image (due to overlapping patches)

Im3d = zeros(aa*p,aa);
for i=1:p
    Im3d(1+(i-1)*aa:i*aa,:) = Im_3d(:,:,i);
end

figure(8)

subplot(1,2,1)
imagesc(I_);
colormap gray
axis image;
colorbar
title('Original')

subplot(1,2,2)
imagesc(Im3d);
colormap gray
axis image;
colorbar
title('Reconstructed-3D')

%%%%%%%%%%%%%%%% 2D TL %%%%%%%%%%%%%%%%
[rows,cols] = ind2sub([sqrt(n1),sqrt(n1)],1:n1);

%Creating all image patches (including wrap around patches)
TE = zeros(n1,N2*p);
Ib1 = [I I(:,1:(sqrt(n1)-1),:);I(1:(sqrt(n1)-1),:,:) I(1:(sqrt(n1)-1),1:(sqrt(n1)-1),:)];
for j=1:n1
    TE(j,:) = reshape(Ib1(rows(j):rows(j)+aa-1,cols(j):cols(j)+aa-1,:),aa*aa*p,1);
end

X2d = zeros(size(TE));
for i=1:p
    X1 = W2d*TE(:,1+(i-1)*N2:i*N2);
    [s1]=sort(abs(X1(:)),'descend');
    X2d(:,1+(i-1)*N2:i*N2) = X1.*(abs(X1) >= s1(round(sp*n1*N2))); %initialize sparse codes (for the first TLMRI iteration)
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Transform learning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Create a random permutation for the case when only a subset of patches are used in learning
if(N<N2*p)
    de=randperm(N2*p);
else
    de=1:N2*p;
end

%Transform learning iterations
for pt=1:numiter2d
    Witer2d = W2d;
    
    if(pt==1)
        YH = TE(:,de(1:N)); %Training data
        XH = X2d(:,de(1:N)); %Sparse codes
        l3 = l0*N; l2=l3; %Weights on the Frobenius norm and negative log-determinant terms in transform learning
        [U,S,V]=svd((YH*YH') + (0.5*l3*eye(n1)));
        LL2=(inv(U*(S^(1/2))*V'));  %inverse square root factor used in transform learning
    end
    
    %Transform Update
    [GW,Si,R]=svd(LL2*(YH*XH'));
    sig=diag(Si);
    gamm=(1/2)*(sig + (sqrt((sig.^2) + 2*l2)));
    B=R*(diag(gamm))*GW';
    W2d=B*(LL2);
    
    Werror2d(pt)=norm(Witer2d - W2d,'fro')/norm(W2d,'fro');
    figure (4), plot(Werror2d); title('W2d-itererror');
    drawnow;
    
    %Sparse Coding
    if(pt<numiter2d)
        for i=1:p
            X1=W2d*TE(:,1+(i-1)*N2:i*N2);
            [s1]=sort(abs(X1(:)),'descend');
            XH(:,1+(i-1)*N2:i*N2) = X1.*(abs(X1) >= s1(round(sp*n1*N2)));
        end
    end
end

%%%%%%%%% Sparse Coding and Image Compression with the Learned Transform %%%%%%%%%

for i=1:p
    X1=W2d*TE(:,1+(i-1)*N2:i*N2);
    [s1]=sort(abs(X1(:)),'descend');
    X2d(:,1+(i-1)*N2:i*N2) = X1.*(abs(X1) >= s1(round(sp*n1*N2)));%find all the sparse codes with the updated transform
end

Z2d = W2d\X2d;

IMout1=zeros(size(Ib1));

for j=1:n1
    IMout1(rows(j):rows(j)+aa-1,cols(j):cols(j)+aa-1,:) = ...
        IMout1(rows(j):rows(j)+aa-1,cols(j):cols(j)+aa-1,:) + ...
        reshape(Z2d(j,:),aa,aa,p);
end

Im_2d=IMout1(1:aa,1:bb,:);
Im_2d(1:(sqrt(n1)-1),:,:)= Im_2d(1:(sqrt(n1)-1),:,:)+ IMout1(aa+1:size(IMout1,1),1:bb,:);
Im_2d(:, 1:(sqrt(n1)-1),:)=Im_2d(:, 1:(sqrt(n1)-1),:) + IMout1(1:aa,bb+1:size(IMout1,2),:);
Im_2d(1:(sqrt(n1)-1),1:(sqrt(n1)-1),:)= Im_2d(1:(sqrt(n1)-1),1:(sqrt(n1)-1),:)+ IMout1(aa+1:size(IMout1,1),bb+1:size(IMout1,2),:);

Im_2d = Im_2d/n1; % normalize the image (due to overlapping patches)

Im2d = zeros(aa*p,aa);
for i=1:p
    Im2d(1+(i-1)*aa:i*aa,:) = Im_2d(:,:,i);
end

figure(9)

subplot(1,2,1)
imagesc(I_);
colormap gray
axis image;
colorbar
title('Original')

subplot(1,2,2)
imagesc(Im2d);
colormap gray
axis image;
colorbar
title('Reconstructed-2D')

%%%%%%%%%%% Performance Comparison %%%%%%%%%%%

for i=1:p
PSNR2d(i) = psnr(I(:,:,i),indexer(Im2d,i,1,aa));
SSIM2d(i) = ssim(I(:,:,i),indexer(Im2d,i,1,aa));
PSNR3d(i) = psnr(I(:,:,i),indexer(Im3d,i,1,aa));
SSIM3d(i) = ssim(I(:,:,i),indexer(Im3d,i,1,aa));
end

[PSNR2d' SSIM2d' PSNR3d' SSIM3d']
