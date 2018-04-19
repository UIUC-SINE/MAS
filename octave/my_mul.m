% by Fatih Çaðatay Akyön & Ulaþ Kamacý 2017
% fatihcagatayakyon@gmail.com & kamaci.ulas@gmail.com

function r=my_mul(x,y,aa) 
% multiplies 2 block matrices

% r is the result of multiplication of matrices x and y, aa is the bloc size block

s1=size(x,1)/aa;
s2=size(x,2)/aa;
s3=size(y,2)/aa;

% r = x.y : [s1 x s2].[s2 x s3] = [s1 x s3]  blocks ;

r=zeros(size(x,1),size(y,2)); %set the dimension of the output

for i=1:s1
    for j=1:s3
        for k=1:s2            
            r(1+(i-1)*aa : i*aa , 1+(j-1)*aa : j*aa) =  ... 
            r(1+(i-1)*aa : i*aa , 1+(j-1)*aa : j*aa) +  ...
            x(1+(i-1)*aa : i*aa , 1+(k-1)*aa : k*aa) .* ...
            y(1+(k-1)*aa : k*aa , 1+(j-1)*aa : j*aa) ;
        end
    end
end

end