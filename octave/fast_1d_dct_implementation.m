%% 1D DCT Implementation - For Loop vs Reshaped Matrix Multiplication

aa = 128;
p = 6;

Im = rand(p*aa,aa);

tic
for kp=1:10
    for i = 1:p
        X1(1+(i-1)*aa:i*aa, 1:aa) = dct2(indexer(Im,i,1,aa)); % calculate 2D DCT of each image
    end
    
    for i = 1:aa %DCT in the wavelength dim.
        for j = 1:aa
            X1(i:aa:end,j) = dct(X1(i:aa:end,j));
            %Note: since the matrix X is 2 dimensional, we catch the
            %wavelength dimension by jumping aa amount downwards. Have a
            %look at the image "wavelength_trajectory_illustration.png"
        end
    end
end
toc

clear X

tic
TE = zeros(p,aa*aa);
for kp=1:10
    
    for i = 1:p
        TE(i,:) = reshape(dct2(indexer(Im,i,1,aa)),aa*aa,1); % calculate 2D DCT of each image
    end
    
    XH = dctmtx(p)*TE;
    
    for i=1:p
        X2(1+(i-1)*aa:i*aa, 1:aa) = reshape(XH(i,:),aa,aa);
    end
    
end
toc

norm(X1(:)-X2(:))

%% inv DCT

aa = 128;
p = 6;

X = rand(p*aa,aa);

tic
for kp=1:10
c = block_idct2(X,aa);
for i = 1:aa %invDCT in the wavelength dim.
    for j = 1:aa
        c(i:aa:end,j) = idct(c(i:aa:end,j));
    end
end
end
toc

tic
for kp=1:10
for i = 1:p
    TE(i,:) = reshape(idct2(indexer(X,i,1,aa)),aa*aa,1); % calculate 2D DCT of each image
end

XH = dctmtx(p)'*TE;

for i=1:p
    c2(1+(i-1)*aa:i*aa, 1:aa) = reshape(XH(i,:),aa,aa);
end
end
toc

norm(c(:)-c2(:))


