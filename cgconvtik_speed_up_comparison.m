%% CG Convtik Speed Up Trial

P = reshape(p,sizeIx,sizeIy);

tic
for t=1:500
%First compute b= H*C* p
B=zeros(sizeOx,sizeOy);
for k=1:K
    for s=1:S
        Hks=H((k-1)*N1+1:k*N1,(s-1)*N2+1:s*N2);
        Xs= Code .* P((s-1)*L1+1:s*L1,:);
        B((k-1)*L1+1:k*L1,:)=B((k-1)*L1+1:k*L1,:)+cconv2(Xs,Hks,L1,L2);
    end
end

%Then compute C'*Wc* b
Bweighted=Wc.*B;
Tmp1=zeros(sizeIx,sizeIy);
for sPrime=1:S
    s=S-sPrime+1;
    for kPrime=1:K
        %inverted blurring kernel for k'=K-k+1, s'=P-s+1
        Hks=Hreflected((kPrime-1)*N1+1:kPrime*N1,(sPrime-1)*N2+1:sPrime*N2);
        k=K-kPrime+1;
        Bk=Bweighted((k-1)*L1+1:k*L1,:);
        Tmp1((s-1)*L1+1:s*L1,:)=Tmp1((s-1)*L1+1:s*L1,:)+cconv2(Bk,Hks,L1,L2);
    end
end
Tmp1 = Code_mtx .* Tmp1;
Tmp1 = mu * Tmp1;
end
toc

psfsize = 51;
aa = 128;

gamma_H = block_fft2(H, psfsize, aa);
gamma_H = real(gamma_H);
gamma_ = my_mul(my_herm(gamma_H, aa), gamma_H, aa);

tic
for t=1:500
Tmp2 = Code_mtx .* block_ifft2(my_mul(gamma_,block_fft2(Code_mtx .* P,aa,aa),aa),aa);
Tmp2 = mu * Tmp2;
end
toc