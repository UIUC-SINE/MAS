% by Fatih Çaðatay Akyön & Ulaþ Kamacý 2017
% fatihcagatayakyon@gmail.com & kamaci.ulas@gmail.com

function y=inv2(x,aa)
% takes the inverse of a block matrix with block sizes of aa by aa
% use the function my_inv, instead

y=zeros(size(x,1),size(x,2));

y(1:aa , 1:aa)=1./(x(1:aa , 1:aa)-x(1:aa , aa+1:2*aa)./x(aa+1:2*aa , aa+1:2*aa).*x(aa+1:2*aa , 1:aa));
y(aa+1:2*aa , aa+1:2*aa)=1./(x(aa+1:2*aa , aa+1:2*aa)-x(aa+1:2*aa , 1:aa)./x(1:aa , 1:aa).*x(1:aa , aa+1:2*aa));
y(1:aa , aa+1:2*aa)=-y(1:aa , 1:aa).*x(1:aa , aa+1:2*aa)./x(aa+1:2*aa , aa+1:2*aa);
y(aa+1:2*aa , 1:aa)=-y(aa+1:2*aa , aa+1:2*aa).*x(aa+1:2*aa , 1:aa)./x(1:aa , 1:aa);

end