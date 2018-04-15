pkg load communications

F1d = dftmtx(5);
F2d = kron(F1d,F1d);

D1d = [1 2 1 0 0; 0 1 2 1 0; 0 0 1 2 1; 1 0 0 1 2; 2 1 0 0 1]';
D2d_col = kron(eye(5),D1d);
D2d_row = kron(D1d,eye(5));

gam1d = F1d*D1d*F1d'/5;

gam2d_col = F2d*D2d_col*F2d'/25;
gam2d_row = F2d*D2d_row*F2d'/25;

figure(1)
imshow(abs(gam1d));
title('gam1d');

figure(2)
imshow(abs(gam2d_col));
title('gam2d\_col');

figure(3)
imshow(abs(gam2d_row));
title('gam2d\_row');
