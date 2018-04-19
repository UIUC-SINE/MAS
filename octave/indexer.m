% by Fatih Çaðatay Akyön & Ulaþ Kamacý 2017
% fatihcagatayakyon@gmail.com & kamaci.ulas@gmail.com

function y=indexer(x,i,j,aa)
% takes the specified block of a block matrix as output

% y is the i by j'th block of block matrix x
% y is a square matrix with size aa by aa

y = x(1+(i-1)*aa:i*aa , 1+(j-1)*aa:j*aa); 
end