function [ C ] = sparse_coding( Im, aa, bb, p, et, kp )

%Sparse Coding
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

TE = zeros(p,aa*bb);
for i = 1:p
    TE(i,:) = reshape(dct2(indexer(Im,i,1,aa)),aa*bb,1); % calculate 2D DCT of each image
end

XH = dctmtx(p)*TE;

for i=1:p
    X(1+(i-1)*aa:i*aa, 1:bb) = reshape(XH(i,:),aa,bb);
end
% Together with the DCT in the wavelength dim., we performed 3D DCT to Im.
% To complete sparse coding, we will only keep the largest s % of the
% coefficients, and make the others zero:

[s] = sort(abs(X(:)),'descend');
X = X.*(abs(X) >= s(round(et(kp)*aa*bb*p)));
% Only largest s fraction of transform domain coeffs (among all the
% images) will remain, rest are zeroed

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Image Update
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% inverse 3D DCT %/%/%/%/%/%/%/%/%/%/%/%/%/%/%/%/

for i = 1:p
    TE(i,:) = reshape(idct2(indexer(X,i,1,aa)),aa*bb,1); % calculate 2D DCT of each image
end

XH = dctmtx(p)'*TE;

for i=1:p
    C(1+(i-1)*aa:i*aa, 1:bb) = reshape(XH(i,:),aa,bb);
end
