% by Fatih Çaðatay Akyön & Ulaþ Kamacý 2017
% fatihcagatayakyon@gmail.com & kamaci.ulas@gmail.com

function [outputImage, w_1] = imposeNoise(inputImage, SNR)
% adds "SNR" dB noise onto the image

noiseVariance = var(inputImage(:))/(10^(SNR/10));
w_1=sqrt(noiseVariance)*randn(size(inputImage, 1), size(inputImage, 2));
outputImage = inputImage + w_1;
end

