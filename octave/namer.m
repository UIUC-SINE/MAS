% by Fatih Çaðatay Akyön & Ulaþ Kamacý 2017
% fatihcagatayakyon@gmail.com & kamaci.ulas@gmail.com

function zout = namer(t,i,j)
% creates a string consisting of ["string" t, "number" i, and "number" j]

if nargin==3
    zout = [t,num2str(i),num2str(j)];
else
    zout = [t,num2str(i)];
end
end