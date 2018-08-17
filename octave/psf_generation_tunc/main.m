clear variables;
close all;
clc;

%% Photon sieve generator/analyzer main code.
% Written by: Tunc Alkanat - METU - talkanat@gmail.com

% Description: Generates a photon sieve, samples its aperture function and
% calculates its PSF for varying measurement plane - photon sieve plane
% separation.

%% Sieve design parameters.
SIEVE_f = 3.7425;%(m). Desired focal length of the design. Only one of the
% focal length and sieve diameter can be set. If SIEVE_D is defined, overrides SIEVE_f.
SIEVE_D = 10e-3; %(m). Diameter of the photon sieve design. Overrides SIEVE_f if defined.
TARGET_WL = 33.4e-9; %(m). Center wavelength of the design.
MIN_FAB_STRUCTURE = 7.56e-6*4; %(m). Minimum fabricable structure.
HOLE_DIAM_TO_ZONE_WIDTH = 1.53096; % Ratio of pinhole diameter to the width of its corresponding underlying Fresnel zone.
RATIO_OF_OPEN_AREA = 0.6; % Higher value corresponds to a higher count of pinholes per white zone.
ROUND_TO_4 = 0; % Control parameter. Rounds the number of pinholes in each zone to the nearest multiple of four if set.
RANDOMIZE_PINHOLE_START_DEGREE = 1; % Control parameter. Randomizes the radial placement of each zone of generated photon sieve if set.
DOF = 2*MIN_FAB_STRUCTURE^2 / TARGET_WL

%% PSF calculation parameters.
CALCULATE_PSF = 1; % Control parameter. Calculates and saves the PSF if set.
DOF_span = [-7.4 -6 -3.7 -1.4 0]; % DOF span of PSF calculation. e.g. for [-1 1], PSF for f - 1DOF and f + 1DOF will be calculated.
## DOF_span = linspace(-30, 10, 301);
## DOF_span = [0];
DOF_span = linspace(-30, 10, 151);
HORIZONTAL_SAMPLING_CNT = 151; %(pxls - BETTER BE ODD). Number of samples of the aperture function of photon sieve in x dimension. Used for PSF calculation.
VERTICAL_SAMPLING_CNT =  151; %(pxls - BETTER BE ODD). Number of samples of the aperture function of photon sieve in y dimension. Used for PSF calculation.
## SAMPLING_INTERVAL = 4.65e-6; %(m). Separation between samples of same dimension. Used for PSF calculation.
SAMPLING_INTERVAL = TARGET_WL * (SIEVE_f + DOF_span(1) * DOF) / SIEVE_D; %(m). Separation between samples of same dimension. Used for PSF calculation.

%% Space domain sampling parameters.
SAMPLING_SIZE_X_PX = 1920; %(px). Number of samples of the aperture function of photon sieve in x dimension.
SAMPLING_SIZE_Y_PX = 1080; %(px). Number of samples of the aperture function of photon sieve in y dimension.
PIXEL_PITCH = 7.56e-6; %(m). Separation between samples of same dimension.

%% Include/output dirs.
outPath = '.\output'; % Relative directory of output folder.
addpath('includes'); % Include scripts.

%% Designs the photon sieve.
if exist('SIEVE_D', 'var')
    % f is the free parameter. Generates photon sieve design. Stores design
    % data in optimalSieveStruct.
    optimalSieveStruct = getOptimalSieveData_diam(SIEVE_D,MIN_FAB_STRUCTURE, TARGET_WL, HOLE_DIAM_TO_ZONE_WIDTH, RATIO_OF_OPEN_AREA, ROUND_TO_4, RANDOMIZE_PINHOLE_START_DEGREE);
else
    % D is the free parameter.
    optimalSieveStruct = getOptimalSieveData_focus(SIEVE_f,MIN_FAB_STRUCTURE, TARGET_WL, HOLE_DIAM_TO_ZONE_WIDTH, RATIO_OF_OPEN_AREA, ROUND_TO_4, RANDOMIZE_PINHOLE_START_DEGREE);
end

printf('Done calculate optimalSieveStruct\n');
fflush(stdout);


%% Optimal Sieve Struct Fields
% Sieve Data: 1-by-1 cell(Cell dimension reserved for further use). The
% cell contains a 1-by-NumberOfZones cell. Each cell includes
% a matrix of dimensions 3-by-NumberOfHoles. Where each column specifies a pinhole. [centerCoordX; centerCoordY; diameter] 
% All data is in meters.
% f: Focal length of photon sieve.
% d: Diameter of photon sieve.
% N: Number of 'white' zones of the design.
% N_h: Vector of 1-by-NumberOfWhiteZones. Includes number of pinholes of
% each zone.
% TARGET_WL
% MIN_FAB_STRUCTURE

% Construct the name of the output folder. Is of form:
% <DATE>_LAMBDA_<TARGET_WL_IN_NM>_f_<FOCAL_LENGTH_IN_M>
out = [outPath, '\', datestr(now, 30) '_LAMBDA_' num2str(TARGET_WL*1e9) '_f_' num2str(optimalSieveStruct.f)];
mkdir(out); % Make the output dir.
save([out '\optimalSieveData.mat'], 'optimalSieveStruct'); % Save the optimal sieve struct.

%Get aperture function of the constructed sieve. sampledSieveStruct
%includes aperture function, x and y grids of sampling 2-D impulse train.
## sampledSieveStruct = getSampledApertureFunction(SAMPLING_SIZE_X_PX, SAMPLING_SIZE_Y_PX, PIXEL_PITCH, optimalSieveStruct); 
## save([out '\sampledSieve.mat'], 'sampledSieveStruct'); %saves as .mat.
## imwrite(sampledSieveStruct.a, [out '\apertureFunction.png']); %saves as image.

%% Writes photon sieve properties to .txt file.
fileID = fopen([out '\SieveProperties.txt'], 'wt');
fprintf(fileID, '*********DATA*********\n');
fprintf(fileID, 'D = %f mm\n', optimalSieveStruct.D*1e3);
fprintf(fileID, 'LAMBDA = %f nm\n', optimalSieveStruct.TARGET_WL*1e9);
fprintf(fileID, 'DELTA = %f um\n', optimalSieveStruct.MIN_FAB_STRUCTURE*1e6);
fprintf(fileID, 'F = %f m\n', optimalSieveStruct.f);
fprintf(fileID, 'DOF = %f mm\n', 2*(optimalSieveStruct.MIN_FAB_STRUCTURE^2)/optimalSieveStruct.TARGET_WL*1e3); % Depth pf Focus.
fprintf(fileID, 'N_zones = %d\n', optimalSieveStruct.N);
fprintf(fileID, 'Rayleigh Resolution = %f radians = %f degrees\n',...
    1.21966989*optimalSieveStruct.TARGET_WL/optimalSieveStruct.D,...
    rad2deg(1.21966989*optimalSieveStruct.TARGET_WL/optimalSieveStruct.D)); % Rayleigh resolution of design.
fprintf(fileID, '******END*OF*DATA******\n');


fprintf(fileID, '\n*********DATA2*********\n');
fprintf(fileID, 'HOLE_DIAM_TO_ZONE_WIDTH = %f\n', HOLE_DIAM_TO_ZONE_WIDTH);
fprintf(fileID, 'RATIO_OF_OPEN_AREA = %f\n', RATIO_OF_OPEN_AREA);
fprintf(fileID, 'ROUND_TO_4 = %d\n', ROUND_TO_4);
fprintf(fileID, 'MAX_SIZE_X_PX = %d\n', SAMPLING_SIZE_X_PX);
fprintf(fileID, 'MAX_SIZE_Y_PX = %d\n', SAMPLING_SIZE_Y_PX);
fprintf(fileID, 'PIXEL_PITCH = %f um\n', PIXEL_PITCH*1e6);
fprintf(fileID, '******END*OF*DATA2******\n');

fclose(fileID);
fprintf('End of design. \n');

%% PSF Calculation
if CALCULATE_PSF
    fprintf('Start of PSF calculation. \n');
    [g_squared, xGrid, yGrid] = calculate2DPSF(optimalSieveStruct, DOF_span, HORIZONTAL_SAMPLING_CNT, VERTICAL_SAMPLING_CNT, SAMPLING_INTERVAL, out);
    fprintf('End of PSF calculation. \n');
end

slices = zeros(151, 0)

for i = 1:length(g_squared)
  slice = fftshift(fft2(g_squared{i}(:, 75)));
  slices = cat(2, slices, slice);
endfor

% Function write2DPSF_FSO returns the squared abs of the calculated PSF for
% each DOF_span value along with the space domain sampling points in both
% directions.
