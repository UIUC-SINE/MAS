%% 2D and 3D Various Transforms Compressibility tests
% Report counterparts of the symbols used here:
% Im: x, aa: N, p: S, sp: \beta, X2: b, X3: c, X2t: \hat b, X3t: \hat c

aa = 128;
bb = 128;
p = 6;
load('reflectances_6_selected.mat');
m0 = 230;
n0 = 630;
I_ = zeros(aa,aa,p);
I = zeros(p*aa,aa);
for j=1:p
    temp = indexer(reflectances_selected,j,1,820);
    I(1+(j-1)*aa:j*aa, 1:aa) = temp(m0:m0+127,n0:n0+127);
    I_(:,:,j) = temp(m0:m0+127,n0:n0+127);
end
I = I/max(I(:));
I_ = I_/max(I_(:));
clear temp;
sp = 0.030;
sc_opt = 'all';

%% 2D and 3D Full DCT 

X2f = zeros(p,aa*bb);
for i = 1:p
    X2f(i,:) = reshape(dct2(indexer(I,i,1,aa)),aa*bb,1); % calculate 2D DCT of each image
end

X3f = dctmtx(p)*X2f;

if strcmp(sc_opt,'all')
    
    [s2f]=sort(abs(X2f(:)),'descend');
    X2ft = X2f.*(abs(X2f) >= s2f(round(sp*aa*aa*p)));
    
elseif strcmp(sc_opt,'single')
    
    X2ft = zeros(size(X2f));
    for i=1:p
        [s2f]=sort(abs(X2f(i,:)),'descend');
        X2ft(i,:) = X2f(i,:).*(abs(X2f(i,:)) >= s2f(round(sp*aa*aa)));
    end
    
end

[s3f]=sort(abs(X3f(:)),'descend');
X3ft = X3f.*(abs(X3f) >= s3f(round(sp*aa*aa*p)));

for i = 1:p
    c2f(1+(i-1)*aa:i*aa, 1:aa) = idct2(reshape(X2ft(i,:),aa,aa)); 
end

TE = zeros(p,aa*aa);
for i = 1:p
    TE(i,:) = reshape(idct2(reshape(X3ft(i,:),aa,aa)),aa*bb,1); 
end

XH = dctmtx(p)'*TE;

for i=1:p
    c3f(1+(i-1)*aa:i*aa, 1:bb) = reshape(XH(i,:),aa,bb);
end

%% 2D and 3D Patch Based DCT

n1 = 81;
n2 = 6;
N2 = aa*aa;
W2pd = kron(dctmtx(sqrt(n1)),dctmtx(sqrt(n1)));
W3pd = kron(dctmtx(n2),kron(dctmtx(sqrt(n1)),dctmtx(sqrt(n1))));

%%%%%%%%%%%%%%%% 3D %%%%%%%%%%%%%%%%
[rows,cols,slices] = ind2sub([sqrt(n1),sqrt(n1),n2],1:n1*n2);

%Creating all image patches (including wrap around patches)
TE3d = zeros(n1*n2,N2*p);
Ibs = cat(3,I_,I_(:,:,1:n2-1));
Ib1 = [Ibs Ibs(:,1:(sqrt(n1)-1),:);Ibs(1:(sqrt(n1)-1),:,:) Ibs(1:(sqrt(n1)-1),1:(sqrt(n1)-1),:)];
for j=1:n1*n2
    TE3d(j,:) = reshape(Ib1(rows(j):rows(j)+aa-1,cols(j):cols(j)+aa-1,slices(j):slices(j)+p-1),aa*aa*p,1);
end

X3pd = W3pd*TE3d;
[s1]=sort(abs(X3pd(:)),'descend');
X3pdt = X3pd.*(abs(X3pd) >= s1(round(sp*n1*n2*N2*p))); %get the sparse code
Z3d = W3pd\X3pdt;

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

c3pd = zeros(aa*p,aa);
for i=1:p
    c3pd(1+(i-1)*aa:i*aa,:) = Im_3d(:,:,i);
end

%%%%%%%%%%%%%%%% 2D %%%%%%%%%%%%%%%%
[rows,cols] = ind2sub([sqrt(n1),sqrt(n1)],1:n1);

%Creating all image patches (including wrap around patches)
TE2d = zeros(n1,N2*p);
Ib1 = [I_ I_(:,1:(sqrt(n1)-1),:);I_(1:(sqrt(n1)-1),:,:) I_(1:(sqrt(n1)-1),1:(sqrt(n1)-1),:)];
for j=1:n1
    TE2d(j,:) = reshape(Ib1(rows(j):rows(j)+aa-1,cols(j):cols(j)+aa-1,:),aa*aa*p,1);
end

if strcmp(sc_opt,'all')
    X2pd = W2pd*TE2d;
    [s1]=sort(abs(X2pd(:)),'descend');
    X2pdt = X2pd.*(abs(X2pd) >= s1(round(sp*n1*N2*p)));
elseif strcmp(sc_opt,'single')
    X2pd = W2pd*TE2d;
    for i=1:p
        temp = X2pd(:,1+(i-1)*N2:i*N2);
        [s1]=sort(abs(temp(:)),'descend');
        X2pdt(:,1+(i-1)*N2:i*N2) = temp.*(abs(temp) >= s1(round(sp*n1*N2)));
    end
end

Z2d = W2pd\X2pdt; 

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

c2pd = zeros(aa*p,aa);
for i=1:p
    c2pd(1+(i-1)*aa:i*aa,:) = Im_2d(:,:,i);
end


%% 2D and 3D Patch Based Learned Transforms

n1 = 81;
n2 = 6;
N2 = aa*aa;
W2pl = W2d;
W3pl = W3d;

%%%%%%%%%%%%%%%% 3D %%%%%%%%%%%%%%%%
[rows,cols,slices] = ind2sub([sqrt(n1),sqrt(n1),n2],1:n1*n2);

%Creating all image patches (including wrap around patches)
TE3d = zeros(n1*n2,N2*p);
Ibs = cat(3,I_,I_(:,:,1:n2-1));
Ib1 = [Ibs Ibs(:,1:(sqrt(n1)-1),:);Ibs(1:(sqrt(n1)-1),:,:) Ibs(1:(sqrt(n1)-1),1:(sqrt(n1)-1),:)];
for j=1:n1*n2
    TE3d(j,:) = reshape(Ib1(rows(j):rows(j)+aa-1,cols(j):cols(j)+aa-1,slices(j):slices(j)+p-1),aa*aa*p,1);
end

X3pl = W3pl*TE3d;
[s1]=sort(abs(X3pl(:)),'descend');
X3plt = X3pl.*(abs(X3pl) >= s1(round(sp*n1*n2*N2*p))); %get the sparse code
Z3d = W3pl\X3plt;

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

c3pl = zeros(aa*p,aa);
for i=1:p
    c3pl(1+(i-1)*aa:i*aa,:) = Im_3d(:,:,i);
end

%%%%%%%%%%%%%%%% 2D %%%%%%%%%%%%%%%%
[rows,cols] = ind2sub([sqrt(n1),sqrt(n1)],1:n1);

%Creating all image patches (including wrap around patches)
TE2d = zeros(n1,N2*p);
Ib1 = [I_ I_(:,1:(sqrt(n1)-1),:);I_(1:(sqrt(n1)-1),:,:) I_(1:(sqrt(n1)-1),1:(sqrt(n1)-1),:)];
for j=1:n1
    TE2d(j,:) = reshape(Ib1(rows(j):rows(j)+aa-1,cols(j):cols(j)+aa-1,:),aa*aa*p,1);
end

if strcmp(sc_opt,'all')
    X2pl = W2pl*TE2d;
    [s1]=sort(abs(X2pl(:)),'descend');
    X2plt = X2pl.*(abs(X2pl) >= s1(round(sp*n1*N2*p)));
elseif strcmp(sc_opt,'single')
    X2pl = W2pl*TE2d;
    for i=1:p
        temp = X2pl(:,1+(i-1)*N2:i*N2);
        [s1]=sort(abs(temp(:)),'descend');
        X2plt(:,1+(i-1)*N2:i*N2) = temp.*(abs(temp) >= s1(round(sp*n1*N2)));
    end
end

Z2d = W2pl\X2plt; 

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

c2pl = zeros(aa*p,aa);
for i=1:p
    c2pl(1+(i-1)*aa:i*aa,:) = Im_2d(:,:,i);
end

%% Sparse Energy Comparison

se2f = ((X2ft(:))'*X2ft(:))/((X2f(:))'*X2f(:));
se3f = ((X3ft(:))'*X3ft(:))/((X3f(:))'*X3f(:));
se2pd = ((X2pdt(:))'*X2pdt(:))/((X2pd(:))'*X2pd(:));
se3pd = ((X3pdt(:))'*X3pdt(:))/((X3pd(:))'*X3pd(:));
se2pl = ((X2plt(:))'*X2plt(:))/((X2pl(:))'*X2pl(:));
se3pl = ((X3plt(:))'*X3plt(:))/((X3pl(:))'*X3pl(:));

SEs = [se2f se3f se2pd se3pd se2pl se3pl]

%% PSNRs SSIMs and Figures

figure(1)

subplot(1,3,1)
imagesc(I);
colormap gray
axis image;
colorbar
title('Original')

subplot(1,3,2)
imagesc(c2f);
colormap gray
axis image;
colorbar
title('2D Full DCT')

subplot(1,3,3)
imagesc(c3f);
colormap gray
axis image;
colorbar
title('3D Full DCT')

figure(2)

subplot(1,3,1)
imagesc(I);
colormap gray
axis image;
colorbar
title('Original')

subplot(1,3,2)
imagesc(c2pd);
colormap gray
axis image;
colorbar
title('2D Patch-Based DCT')

subplot(1,3,3)
imagesc(c3pd);
colormap gray
axis image;
colorbar
title('3D Patch-Based DCT')

figure(3)

subplot(1,3,1)
imagesc(I);
colormap gray
axis image;
colorbar
title('Original')

subplot(1,3,2)
imagesc(c2pl);
colormap gray
axis image;
colorbar
title('2D Patch-Based Learned')

subplot(1,3,3)
imagesc(c3pl);
colormap gray
axis image;
colorbar
title('3D Patch-Based Learned')

for i = 1:p
    psnrs2f(i) = psnr(indexer(c2f,i,1,aa),indexer(I,i,1,aa));
    psnrs3f(i) = psnr(indexer(c3f,i,1,aa),indexer(I,i,1,aa));
    psnrs2pd(i) = psnr(indexer(c2pd,i,1,aa),indexer(I,i,1,aa));
    psnrs3pd(i) = psnr(indexer(c3pd,i,1,aa),indexer(I,i,1,aa));
    psnrs2pl(i) = psnr(indexer(c2pl,i,1,aa),indexer(I,i,1,aa));
    psnrs3pl(i) = psnr(indexer(c3pl,i,1,aa),indexer(I,i,1,aa));
    ssims2f(i) = ssim(indexer(c2f,i,1,aa),indexer(I,i,1,aa));
    ssims3f(i) = ssim(indexer(c3f,i,1,aa),indexer(I,i,1,aa));
    ssims2pd(i) = ssim(indexer(c2pd,i,1,aa),indexer(I,i,1,aa));
    ssims3pd(i) = ssim(indexer(c3pd,i,1,aa),indexer(I,i,1,aa));
    ssims2pl(i) = ssim(indexer(c2pl,i,1,aa),indexer(I,i,1,aa));
    ssims3pl(i) = ssim(indexer(c3pl,i,1,aa),indexer(I,i,1,aa));
end

psnrs = [psnrs2f' psnrs3f' psnrs2pd' psnrs3pd' psnrs2pl' psnrs3pl']
ssims = [ssims2f' ssims3f' ssims2pd' ssims3pd' ssims2pl' ssims3pl']