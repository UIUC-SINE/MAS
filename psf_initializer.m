k = 3;
p = 3;

%%
% psfs = [incoherentPsf_11, incoherentPsf_12; incoherentPsf_21, incoherentPsf_22];
% psfsize = 51;
%%
load('3measurement3sourcePSFS.mat');
psfs = H;
psfsize = 51;
%%
% psfnum = p*k;

% for a=1:psfnum
% load([num2str(a),'.mat']);
% end

% psfsize = size(psf11,1);

% for i=1:k
%     for j=1:p            
%         psfs(1+(i-1)*psfsize : i*psfsize , 1+(j-1)*psfsize : j*psfsize) =  namer('psf',i,j); 
%     end
% end