function [ pines_ozo_sparse ] = sparse_coding_slow( pines_ozo, aa, bb, p, et, kp )

pines_ozo_2dct = [];
for i = 1:p
    temp = dct2(reshape(pines_ozo(i*aa*bb-aa*bb+1:i*aa*bb),aa,bb));
    pines_ozo_2dct(i*aa*bb-aa*bb+1:i*aa*bb,:) = temp(:);
end



pines_ozo_3dct = zeros(size(pines_ozo_2dct));
for i = 1: aa*bb
    pines_ozo_3dct(i:aa*bb:end,1) = dct(pines_ozo_2dct(i:aa*bb:end,1));
end

[s] = sort(abs(pines_ozo_3dct(:)),'descend');
 pines_ozo_3dct_sparse = pines_ozo_3dct.*(abs(pines_ozo_3dct) >= s(round(et(kp)*aa*bb*p))); 

 pines_ozo_1dct_sparse = [];
 for i = 1:p
    temp = idct2(reshape(pines_ozo_3dct_sparse(i*aa*bb-aa*bb+1:i*aa*bb),aa,bb));
    pines_ozo_1dct_sparse(i*aa*bb-aa*bb+1:i*aa*bb) = temp(:);
 end

 pines_ozo_sparse = zeros(size(pines_ozo_2dct));
 for i = 1: aa*bb
    pines_ozo_sparse(i:aa*bb:end) = idct(pines_ozo_1dct_sparse(i:aa*bb:end));
 end

% pines_ozo_sparse_indexed = [];
% for ii = 1:p
%     temp = pines_ozo_sparse(ii*aa*bb-aa*bb+1:ii*aa*bb);
%     pines_ozo_sparse_indexed = [pines_ozo_sparse_indexed;reshape(temp,aa,bb)];
% end
