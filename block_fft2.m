% by Fatih Çaðatay Akyön & Ulaþ Kamacý 2017
% fatihcagatayakyon@gmail.com & kamaci.ulas@gmail.com

function fft_x=block_fft2(x,a,aa)
% x: block matrix to take 2d block ffts
% a: block sizes of x
% aa: desired output block sizes

[k,p] = size(x);
k=k/a; p=p/a;

if a==aa
    for i=1:k
        for j=1:p
            y=indexer(x,i,j,a);
            y=fftshift(fft2(ifftshift(y)));
            fft_x(1+(i-1)*aa:i*aa, 1+(j-1)*aa:j*aa) = y;
        end
    end
else
    for i=1:k
        for j=1:p
            y=indexer(x,i,j,a);
            y=padarray(y,[(aa/2-(size(y,1)-1)/2) (aa/2-(size(y,2)-1)/2)]);
            y=y(1:aa,1:aa);
            y=fftshift(fft2(ifftshift(y)));
            fft_x(1+(i-1)*aa:i*aa, 1+(j-1)*aa:j*aa) = y;
        end
    end
end

end