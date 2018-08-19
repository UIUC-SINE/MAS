function [optimalSieveStruct] = getOptimalSieveData_diam(D, MIN_FAB_STRUCTURE, TARGET_WL, HOLE_DIAM_TO_ZONE_WIDTH, RATIO_OF_OPEN_AREA, ROUND_TO_4, RANDOMIZE_PINHOLE_START_DEGREE)
%Compute yielding focal length.
f = D*MIN_FAB_STRUCTURE/TARGET_WL;

%Compute number of white zones.
N = floor(D^2/8/TARGET_WL/f);

%Compute radius vals.
radii = sqrt(2*TARGET_WL*f*(1:N));

%Compute w values.
w = (TARGET_WL*f/2)./radii;

%Compute d values.
d = HOLE_DIAM_TO_ZONE_WIDTH.*w;

%Number of Holes
N_h = round((RATIO_OF_OPEN_AREA*8)*w.*radii./(d.^2));

if ROUND_TO_4
    N_h = round2_4(N_h);
end

if RANDOMIZE_PINHOLE_START_DEGREE
    pinholeStartDegreeOffset = -pi + (2*pi).*rand(1, N);
else
    pinholeStartDegreeOffset = zeros(1, N);
end

% Calculate and store x and y vals.
optimalSieveStruct.sieveData = cell(1);
optimalSieveStruct.sieveData{1} = cell(1, N);
printf('Calculating mask for %d zones', N);
fflush(stdout);
for cZones = 1:N
    if mod(cZones, 10) == 0
        printf('Iterations %d\n', cZones);
        fflush(stdout);
    endif
    optimalSieveStruct.sieveData{1}{cZones} = [zeros(2, N_h(cZones)); d(cZones)*ones(1, N_h(cZones))];
    for cHoles = 1:N_h(cZones)
        optimalSieveStruct.sieveData{1}{cZones}(1, cHoles) =...
            radii(cZones)*...
            cos(2*pi/N_h(cZones)*(cHoles - 1)...
            + pinholeStartDegreeOffset(cZones));
        optimalSieveStruct.sieveData{1}{cZones}(2, cHoles) =...
            radii(cZones)*...
            sin(2*pi/N_h(cZones)*(cHoles - 1)...
            + pinholeStartDegreeOffset(cZones));
    end
end

optimalSieveStruct.f = f;
optimalSieveStruct.D = D;
optimalSieveStruct.N = N;
optimalSieveStruct.N_h = N_h;
optimalSieveStruct.TARGET_WL = TARGET_WL;
optimalSieveStruct.MIN_FAB_STRUCTURE = MIN_FAB_STRUCTURE;

end

