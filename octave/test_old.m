aa=128;
bb=128;
n=36;
wd=1;
s=0.16;
N2=128^2;
W = kron(dctmtx(6),dctmtx(6));
% W = 0.2*randn(36,36);
% W = TV_mtx(36);
% W = eye(36);
% W=sqrt(diag(1./diag(W*W')))*W;
% W=z;
% W=paramsout.transform;

%%%Patching Begins%%%
Ib= [I1 I1(:,1:(sqrt(n)-1));I1(1:(sqrt(n)-1),:) I1(1:(sqrt(n)-1),1:(sqrt(n)-1))];
[TE,idx] = my_im2col(Ib,[sqrt(n),sqrt(n)],wd); %Columns of TE are the patches
[rows,cols] = ind2sub(size(Ib)-sqrt(n)+1,idx);
X=W*TE; %Transform domain coefficients
[sorted_X]=sort(abs(X(:)),'descend');
X_coded = X.*(abs(X) >= sorted_X(round(s*n*N2))); %sparse codes
[new_sorted_X]=sort(abs(X_coded(:)),'descend');
new_sorted_X=new_sorted_X/new_sorted_X(1);
ZZ=[sorted_X/sorted_X(1) new_sorted_X]; %compare them


IMout1=zeros(size(Ib));
bbb=sqrt(n);
ZZ1= (W'*X_coded); %come back to the image domain from the sparse codes

for jj = 1:10000:N2
    jumpSize = min(jj+10000-1,N2);
    block=reshape(ZZ1(:,jj:jumpSize),bbb,bbb,jumpSize-jj+1);
    for ii  = jj:jumpSize
        col = cols(ii); row = rows(ii);
        IMout1(row:row+bbb-1,col:col+bbb-1)=IMout1(row:row+bbb-1,col:col+bbb-1)+block(:,:,ii-jj+1);
    end
end
   
I_coded=zeros(aa,bb);
I_coded(1:aa,1:bb)=IMout1(1:aa,1:bb);
I_coded(1:(sqrt(n)-1),:)= I_coded(1:(sqrt(n)-1),:)+ IMout1(aa+1:size(IMout1,1),1:bb);
I_coded(:, 1:(sqrt(n)-1))=I_coded(:, 1:(sqrt(n)-1)) + IMout1(1:aa,bb+1:size(IMout1,2));
I_coded(1:(sqrt(n)-1),1:(sqrt(n)-1))= I_coded(1:(sqrt(n)-1),1:(sqrt(n)-1))+ IMout1(aa+1:size(IMout1,1),bb+1:size(IMout1,2));
I_coded=I_coded/n; %normalize the unpatched image (because overlapping patches are added)

PSNR_coding=20*log10((sqrt(aa*bb))*1/norm(double(abs(I_coded))-double(abs(I1)),'fro'))
figure(4)
imshow(I_coded);
title(['s=',num2str(s),' PSNR=',num2str(PSNR_coding)]);


%%
% [I5, ~]=imread('SDO5.jpg');
% I5=double(rgb2gray(I5));
% 
% [I6, ~]=imread('SDO6.jpg');
% I6=double(rgb2gray(I6));
% 
% [I7, ~]=imread('SDO7.jpg');
% I7=double(rgb2gray(I7));
% 
% [I8, ~]=imread('SDO8.jpg');
% I8=double(rgb2gray(I8));
% 
% [I9, ~]=imread('SDO9.jpg');
% I9=double(rgb2gray(I9));
% 
% a=I9(487:512,257:512);
% I9(487:512,1:256)=fliplr(a);
% 
% figure, imagesc(I9)
% 
% I1=I1-min(I1(:));
% I1=I1/max(max(I1));
% 
% 













