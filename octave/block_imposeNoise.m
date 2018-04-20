% by Fatih Çaðatay Akyön & Ulaþ Kamacý 2017
% fatihcagatayakyon@gmail.com & kamaci.ulas@gmail.com

function noisy_x = block_imposeNoise(x,aa,SNR)
% x: matrix consisting of blocks
% aa: bloack size
% SNR: desired SNR per block

[k,p] = size(x);
k=k/aa; p=p/aa;

noisy_x = zeros(size(x,1),size(x,2));
for i=1:k
    for j=1:p
        y=indexer(x,i,j,aa);
        noiseVariance = var(y(:))/(10^(SNR/10));
        noise = sqrt(noiseVariance)*randn(size(y,1),size(y,2));
        noisy_y = y + noise;
        noisy_x(1+(i-1)*aa:i*aa, 1+(j-1)*aa:j*aa) = noisy_y; 
    end
end

end

