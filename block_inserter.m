% by Fatih �a�atay Aky�n & Ula� Kamac� 2017
% fatihcagatayakyon@gmail.com & kamaci.ulas@gmail.com

function y = block_inserter(x, y, i, j, aa)
% inserts x into y, i and j are the indices of desired block position
% aa is the block size

y(1+(i-1)*aa:i*aa, 1+(j-1)*aa:j*aa) = x;
end