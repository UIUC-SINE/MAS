function y = cconv2(x,h,N1,N2)
h=padarray(h,[(floor(N1/2)-(size(h,1)-1)/2) (floor(N2/2)-(size(h,2)-1)/2)]);
h=h(1:N1,1:N2);
h=fftshift(fft2(ifftshift(h)));

if (size(x) - [N1 N2]) ~= 0
x=padarray(x,[(floor(N1/2)-(size(x,1)-1)/2) (floor(N2/2)-(size(x,2)-1)/2)]);
x=x(1:N1,1:N2);
end
x=fftshift(fft2(ifftshift(x)));

y = h.*x;
y = fftshift(ifft2(ifftshift(y)));
