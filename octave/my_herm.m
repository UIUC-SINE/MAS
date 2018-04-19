% by Fatih Çaðatay Akyön & Ulaþ Kamacý 2017
% fatihcagatayakyon@gmail.com & kamaci.ulas@gmail.com

function y=my_herm(x,aa)
% takes the hermitian of a block matrice

% y is the hermitian of x, aa is the block size

s1=size(x,1)/aa;
s2=size(x,2)/aa;

y=zeros(size(x,2),size(x,1));

for i=1:s2
    for j=1:s1
        y(1+(i-1)*aa : i*aa , 1+(j-1)*aa : j*aa) =  ...
        conj(x(1+(j-1)*aa : j*aa , 1+(i-1)*aa : i*aa));
    end
end

end