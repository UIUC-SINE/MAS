function [IOut,paramsout]= tldecon_p2d(I,noisy_Im,psfs,paramsin,I_diff)

% 2D Sparsifying Transform regularized PSSI deconvolution with 2D Patches

%Initializing algorithm parameters
psfsize=paramsin.psfsize;
[k,p]=size(psfs);
k=k/psfsize;
p=p/psfsize;
[aa,bb]=size(I);
aa=aa/p;
N2=aa*bb;

f = paramsin.f;
d = paramsin.d;

La2=(paramsin.nu)*(aa*bb*k);
num=paramsin.num;
num2=paramsin.numiter;
n=paramsin.n1;
N=paramsin.N;
et=paramsin.s;
l0=paramsin.lambda0;
W=paramsin.W01;
ct=paramsin.ct;
co=paramsin.co;
my_plot=paramsin.plot;
[rows,cols] = ind2sub([sqrt(n),sqrt(n)],1:n);

% for i=1:k
%     y=indexer(noisy_Im,i,1,aa);
%     sc1 = (max(max(abs(y))));
%     y=y/sc1;
%     Im(1+(i-1)*aa:i*aa, 1:aa) = y; % to start the iterations with an initial image of intensity between [0 1]
% end
Im = zeros(p*aa,bb);

%initializing performance and convergence metrics
if(ct==1)
    ittime=zeros(1,num);PSNRs=zeros(p,num);Werror=zeros(1,num);
    PSNRs_diff=zeros(p,num);SSIMs = zeros(p,num); SSIMs_diff = zeros(p,num);
end

if(co==1)
    obj=zeros(1,num); sp=zeros(p,num);
    reg=zeros(1,num);dfit=zeros(1,num);
    itererror=zeros(p,num);
end


%TLMRI iterations
for kp=1:num
    tic
    Iiter = zeros([aa aa p]);
    for i=1:p
        Iiter(:,:,i) = indexer(Im,i,1,aa);
    end
    Witer=W;
    
    %Creating all image patches (including wrap around patches)
    TE = zeros(n,N2*p);
    Ib1 = [Iiter Iiter(:,1:(sqrt(n)-1),:);Iiter(1:(sqrt(n)-1),:,:) Iiter(1:(sqrt(n)-1),1:(sqrt(n)-1),:)];
    for j=1:n
        TE(j,:) = reshape(Ib1(rows(j):rows(j)+aa-1,cols(j):cols(j)+aa-1,:),aa*aa*p,1);
    end
    
    if kp==1
        [rows2,cols2] = ind2sub(size(Ib1(:,:,1))-sqrt(n)+1,1:aa*aa);
    end
    
    if(kp==1)
        for i=1:p
            X1 = W*TE(:,1+(i-1)*N2:i*N2);
            [s1]=sort(abs(X1(:)),'descend');
            X(:,1+(i-1)*N2:i*N2) = X1.*(abs(X1) >= s1(round(et(1)*n*N2))); %initialize sparse codes (for the first TLMRI iteration)
        end
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
    for pt=1:num2
        
        if(pt==1)
            YH = TE(:,de(1:N)); %Training data
            XH = X(:,de(1:N)); %Sparse codes
            l3 = p*l0*N; l2=l3; %Weights on the Frobenius norm and negative log-determinant terms in transform learning
            [U,S,V]=svd((YH*YH') + (0.5*l3*eye(n)));
            LL2=(inv(U*(S^(1/2))*V'));  %inverse square root factor used in transform learning
        end
        
        %Transform Update
        [GW,Si,R]=svd(LL2*(YH*XH'));
        sig=diag(Si);
        gamm=(1/2)*(sig + (sqrt((sig.^2) + 2*l2)));
        B=R*(diag(gamm))*GW';
        W=B*(LL2);
        
        %Sparse Coding
        if(pt<num2)
            for i=1:p
                X1=W*TE(:,1+(i-1)*N2:i*N2);
                [s1]=sort(abs(X1(:)),'descend');
                XH(:,1+(i-1)*N2:i*N2) = X1.*(abs(X1) >= s1(round(et(kp)*n*N2)));
            end
        end
    end
    
    for i=1:p
        X1=W*TE(:,1+(i-1)*N2:i*N2);
        [s1]=sort(abs(X1(:)),'descend');
        X(:,1+(i-1)*N2:i*N2) = X1.*(abs(X1) >= s1(round(et(kp)*n*N2)));%find all the sparse codes with the updated transform
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Image Update
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%Code to find the spectrum of the matrix G%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if(kp==1)
        ZZ3=zeros(aa,bb);ZZ3(1,1)=1;
        Zb= [ZZ3 ZZ3(:,1:(sqrt(n)-1));ZZ3(1:(sqrt(n)-1),:) ZZ3(1:(sqrt(n)-1),1:(sqrt(n)-1))];
        blZ3 = zeros(n,aa*aa);
        for j=1:n
            blZ3(j,:) = reshape(Zb(rows(j):rows(j)+aa-1,cols(j):cols(j)+aa-1),aa*aa,1);
        end
        bbb=sqrt(n);
    end
    
    blZ= (W'*W)*blZ3;
    IMoutZ=zeros(size(Ib1));
    
    qp=sum(abs(blZ).^2); inp=find(qp>0);
    blZ2=blZ(:,inp);
    for jj = 1:length(inp)
        ii=inp(jj);
        col = cols2(ii); row = rows2(ii);
        block =reshape(blZ2(:,jj),[bbb,bbb]);
        IMoutZ(row:row+bbb-1,col:col+bbb-1)=IMoutZ(row:row+bbb-1,col:col+bbb-1)+block;
    end
    
    IMout2Z=IMoutZ(1:aa,1:bb);
    IMout2Z(1:(sqrt(n)-1),:)= IMout2Z(1:(sqrt(n)-1),:)+ IMoutZ(aa+1:size(IMoutZ,1),1:bb);
    IMout2Z(:, 1:(sqrt(n)-1))=IMout2Z(:, 1:(sqrt(n)-1)) + IMoutZ(1:aa,bb+1:size(IMoutZ,2));
    IMout2Z(1:(sqrt(n)-1),1:(sqrt(n)-1))= IMout2Z(1:(sqrt(n)-1),1:(sqrt(n)-1))+ IMoutZ(aa+1:size(IMoutZ,1),bb+1:size(IMoutZ,2));
    
    Lb3=fft2(IMout2Z); Lb1=real(Lb3);
    Lb1=fftshift(Lb1);
    %     Lb=Lb*256;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Find the P'W'b1j term %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    FPhWhB = zeros(p*aa,aa);
    IMout1=zeros(size(Ib1));
    ZZ1= W'*X;
    
    for j=1:n
        IMout1(rows(j):rows(j)+aa-1,cols(j):cols(j)+aa-1,:) = ...
            IMout1(rows(j):rows(j)+aa-1,cols(j):cols(j)+aa-1,:) + ...
            reshape(ZZ1(j,:),aa,aa,p);
    end
    
    IMout21=IMout1(1:aa,1:bb,:);
    IMout21(1:(sqrt(n)-1),:,:)= IMout21(1:(sqrt(n)-1),:,:)+ IMout1(aa+1:size(IMout1,1),1:bb,:);
    IMout21(:, 1:(sqrt(n)-1),:)=IMout21(:, 1:(sqrt(n)-1),:) + IMout1(1:aa,bb+1:size(IMout1,2),:);
    IMout21(1:(sqrt(n)-1),1:(sqrt(n)-1),:)= IMout21(1:(sqrt(n)-1),1:(sqrt(n)-1),:)+ IMout1(aa+1:size(IMout1,1),bb+1:size(IMout1,2),:);
    
    for i=1:p
        I31=fftshift(fft2(ifftshift(IMout21(:,:,i))));
        FPhWhB(1+(i-1)*aa:i*aa,:) = I31;
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Image update step %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    Lb = zeros(aa*p,aa*p);
    for i=1:p
        j = i;
        Lb = block_inserter(Lb1, Lb, i, j, aa);
    end

    if(kp==1)
        gamma_H = block_fft2(psfs, psfsize, aa);
        gamma_H = real(gamma_H);
        gamma_ = La2*my_mul(my_herm(gamma_H, aa), gamma_H, aa);
        
        f_Ip = block_fft2(noisy_Im,aa,aa);
        FHhy = La2*my_mul(my_herm(gamma_H, aa), f_Ip, aa);
    end
    
    gamma = gamma_ + Lb;
    
    F_Im = my_mul(my_inv(gamma,p),FHhy+FPhWhB,aa);
    Im = block_ifft2(F_Im,aa);
    
    time=toc;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    %Compute various performance or convergence metrics
    if(ct==1)
        Werror(kp)=norm(Witer - W,'fro')/norm(W,'fro');
        ittime(kp)=time;
        
        for i=1:p
            Iim=indexer(Im,i,1,aa); Ii=indexer(I,i,1,aa); I_diffi=indexer(I_diff,i,1,aa);
            
            PSNRs(i,kp)=20*log10((sqrt(aa*bb))*1/norm(double(abs(Iim))-double(abs(Ii)),'fro'));
            PSNRs_diff(i,kp)=20*log10((sqrt(aa*bb))*1/norm(double(abs(Iim))-double(abs(I_diffi)),'fro'));
            SSIMs(i,kp)=ssim(Iim,Ii);
            SSIMs_diff(i,kp)=ssim(Iim,I_diffi);
            
            itererror(i,kp) = norm(squeeze(Iiter(:,:,i)) - Iim,'fro')/norm(Iim,'fro');
        end
    end
    
    if(co==1)
        
        for i=1:p
            I1m = indexer(Im,i,1,aa);
            Ib1 = [I1m I1m(:,1:(sqrt(n)-1));I1m(1:(sqrt(n)-1),:) I1m(1:(sqrt(n)-1),1:(sqrt(n)-1))];
            for j=1:n
                TE(j,1+(i-1)*N2:i*N2) = reshape(Ib1(rows(j):rows(j)+aa-1,cols(j):cols(j)+aa-1),aa*aa,1);
            end
            sp(i,kp) = (1/N2)*(norm((W*TE) - X(:,1+(i-1)*N2:i*N2),'fro'))^2; %normalized
        end
        
        dfit(1,kp) = (La2/(aa*bb*k))*((norm(noisy_Im-block_ifft2(my_mul(gamma_H,F_Im,aa),aa),'fro'))^2);
        reg(kp)= l0*(-log(abs(det(W))) + 0.5*((norm(W,'fro'))^2));
        obj(kp)= k*dfit(1,kp) + sum(sp(:,kp)) + p*reg(kp);
    end
    
    %real time plot details
    if my_plot==1 && mod(kp,10)==0
        %close(figure(6));
        figure(1)
        clf
        for i=1:p
            plot(SSIMs_diff(i,:));
            hold on;
        end
        
        bas = ['\nu=',num2str(La2),' s=',num2str(et(1))];
        x=' SSIM';
        son=[];
        for i=1:p
            son = [son, x, num2str(i), '=', num2str(SSIMs_diff(i,kp))];
        end
        title({[bas],[son]});
        
        grid minor;
        
        Legend = cell(p,1);
        for i=1:p
            Legend{i}=['SSIM',num2str(i)];
        end
        legend(Legend,'Location','southeast')
        
        figure(2)
        
        subplot(1,2,1)
        imagesc(I);
        colormap gray
        axis image;
        colorbar
        title('Original')
        
        subplot(1,2,2)
        imagesc(Im);
        colormap gray
        axis image;
        colorbar
        title('Reconstructed')
        
%         figure(3), plot(Werror), title(['Transform IterError =',num2str(Werror(kp))]);
        
        drawnow;
        
        if co==1
            figure(4), plot(dfit), title(['Data Fidelity =',num2str(dfit(kp))]);
            figure(5), plot(reg), title(['Transform Regularizer =',num2str(reg(kp))]);
            
            figure(6)
            clf
            for i=1:p
                plot(sp(i,:));
                hold on;
            end
            
            bas = ['\nu=',num2str(La2),' s=',num2str(et(1))];
            x=' sp';
            son=[];
            for i=1:p
                son = [son, x, num2str(i), '=', num2str(sp(i,kp))];
            end
            title({[bas],[son]});
            
            grid minor;
            
            Legend = cell(p,1);
            for i=1:p
                Legend{i}=['sp',num2str(i)];
            end
            legend(Legend,'Location','southeast')
            
            figure(7), plot(obj(1:kp)), title(['Objective Function =',num2str(obj(kp))]);
            
            drawnow;
        end
    end
    
    %    clear I1 I2 I3 I1m I2m I3m
    
    %     figure(1), imshow(Im); title(['Iteration=',num2str(kp),' PSNR=',num2str(PSNR1(kp))]);
    %     figure(2), imshow(I2m); title(['Iteration=',num2str(kp),' PSNR=',num2str(PSNR2(kp))]);
    %     figure(3), imagesc(W1); title('W1'); colorbar
    %     figure(4), imagesc(W2); title('W2'); colorbar
    %     figure(5), plot(sqrt(itererror1)); title(['itererror = ',num2str(itererror1(kp))])
    %     close(figure(6));
    %     figure(6), plot(PSNR1); hold on; plot(PSNR2); plot(PSNR3); title(['\nu=',num2str(La2),' s=',num2str(et(1)),' \lambda=',num2str(l0),' PSNR1=',num2str(PSNR1(kp)),' PSNR2=',num2str(PSNR2(kp)),' PSNR3=',num2str(PSNR3(kp))]); grid minor; legend('PSNR1','PSNR2','PSNR3','Location','southeast');
    %     figure(7), plot(-abs(Imean1-mean(mean(I1)))); title(['mean shift = ',num2str(Imean1(kp)-mean(mean(I1)))])
    %     figure(8), imagesc(Lb1); title('Lb1'); colorbar
    %     figure(9), imagesc(Lb2); title('Lb2'); colorbar
    %     figure(10), imagesc(abs(gi11)); title('gi11'); colorbar
    %     figure(11), imagesc(abs(gi12)); title('gi12'); colorbar
    %     figure(12), imagesc(abs(gi21)); title('gi21'); colorbar
    %     figure(13), imagesc(abs(gi22)); title('gi22'); colorbar
    %     figure(5), imagesc(real(Lb1)); colorbar;
    %     figure(6), imagesc(real(Lb2)); colorbar;
    %     title(num2str(PSNR1(kp)));
    %     drawnow;
    
    %     if(kp>100)
    %     x=1<0;
    %         for j=1:30
    %             x=x||((PSNR1(kp-j+1)>PSNR1(kp-j)) && (PSNR2(kp-j+1)>PSNR2(kp-j)));
    %         end
    %     else
    %         x=1>0;
    %     end
    
    %     if ( min(itererror(:,kp))<3e-4 ) || ((kp > 60) && (PSNR(1,kp)<16)) || ((kp > 3) && (PSNR(1,kp)<5)) || ((kp > 100) && (PSNR(1,kp)<21))
    %         break
    %     end
    
%     stopping criterion 2
        if(min(itererror(:,kp)<5e-4))
            break
        end
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Outputs
IOut = Im;
paramsout.transform=W;
if(ct==1)
    paramsout.Werror=Werror;
    paramsout.PSNRs_last=PSNRs(:,kp);
    paramsout.PSNRs_diff_last=PSNRs_diff(:,kp);
    paramsout.PSNRs=PSNRs;
    paramsout.PSNRs_diff=PSNRs_diff;
    paramsout.SSIMs_last=SSIMs(:,kp);
    paramsout.SSIMs_diff_last=SSIMs_diff(:,kp);
    paramsout.SSIMs=SSIMs;
    paramsout.SSIMs_diff=SSIMs_diff;
    paramsout.itererror=itererror;
    paramsout.runtime=sum(ittime);
end

if(co==1)
    paramsout.obj=obj;
    paramsout.sp=sp;
    paramsout.reg=reg;
    paramsout.dfit=dfit;
end