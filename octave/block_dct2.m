function dct_x=block_dct2(x,a)
%%x: block matrix to take 2d block dcts
%a: block sizes of x
[k,p] = size(x);
k=k/a; p=p/a;

for i=1:k
    for j=1:p
        y=indexer(x,i,j,a);
        y=dct2(y);
        dct_x(1+(i-1)*a:i*a, 1+(j-1)*a:j*a) = y;
    end
end


end