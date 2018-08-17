function [noh] = round2_4(noh)
%ROUND2_4 Summary of this function goes here
%   Detailed explanation goes here
divs = noh./4;
divs = floor(divs);
noh = divs.*4;

end

