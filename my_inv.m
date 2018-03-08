% by Fatih Çaðatay Akyön & Ulaþ Kamacý 2017
% fatihcagatayakyon@gmail.com & kamaci.ulas@gmail.com

function B = my_inv(A,p) 
% takes the inverse of block matrices in an efficient way
% A is the matrix to be inverted, p is number of sources, B is the inverse

if p==1
    B = 1./A;
    
elseif p==2
    p1 = floor(p/2);
    %p2 = ceil(p/2);
    aa = size(A,1)/p;
    
    A11 = A(1:aa*p1,1:aa*p1);
    A12 = A(1:aa*p1,aa*p1+1:aa*p);
    A21 = A(aa*p1+1:aa*p,1:aa*p1);
    A22 = A(aa*p1+1:aa*p,aa*p1+1:aa*p);
    
    A11i = 1./A11;
    A22hat = A22 - my_mul(my_herm(A12,aa),my_mul(A11i,A12,aa),aa);
    A22hati = 1./A22hat;
    
    A11hati = A11i + my_mul(A11i,my_mul(A12,my_mul(A22hati,my_mul(my_herm(A12,aa),A11i,aa),aa),aa),aa);
    
    A21hat = -my_mul(A22hati,my_mul(A21,A11i,aa),aa);
    A12hat = my_herm(A21hat,aa);
    
    B = [A11hati A12hat; A21hat A22hati];
    
else
    p1 = floor(p/2);
    p2 = ceil(p/2);
    aa = size(A,1)/p;
    
    A11 = A(1:aa*p1,1:aa*p1);
    A12 = A(1:aa*p1,aa*p1+1:aa*p);
    A21 = A(aa*p1+1:aa*p,1:aa*p1);
    A22 = A(aa*p1+1:aa*p,aa*p1+1:aa*p);
    
    A11i = my_inv(A11,p1);
    A22hat = A22 - my_mul(my_herm(A12,aa),my_mul(A11i,A12,aa),aa);
    A22hati = my_inv(A22hat,p2);
    
    A11hati = A11i + my_mul(A11i,my_mul(A12,my_mul(A22hati,my_mul(my_herm(A12,aa),A11i,aa),aa),aa),aa);
    
    A21hat = -my_mul(A22hati,my_mul(A21,A11i,aa),aa);
    A12hat = my_herm(A21hat,aa);
    
    B = [A11hati A12hat; A21hat A22hati];
end
        