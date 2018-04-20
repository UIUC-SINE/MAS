function idct_x=block_idct2(x, aa)
%%x: block matrix to take 2d block iffts
%aa: block sizes
[k,p] = size(x);
k=k/aa; p=p/aa;

for i=1:k
    for j=1:p
        y=indexer(x,i,j,aa);
        y=idct2(y);
        idct_x(1+(i-1)*aa:i*aa, 1+(j-1)*aa:j*aa) = y;
    end
end

end