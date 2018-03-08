% by Fatih Çaðatay Akyön & Ulaþ Kamacý 2017
% fatihcagatayakyon@gmail.com & kamaci.ulas@gmail.com

function ifft_x=block_ifft2(x, aa)
% x: block matrix to take 2d block iffts
% aa: block sizes

[k,p] = size(x);
k=k/aa; p=p/aa;

for i=1:k
    for j=1:p
        y=indexer(x,i,j,aa);
        y=fftshift(ifft2(ifftshift(y)));
        ifft_x(1+(i-1)*aa:i*aa, 1+(j-1)*aa:j*aa) = y;
    end
end

end