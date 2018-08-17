function [sampledSieveStruct] = getSampledApertureFunction(MAX_SIZE_X_PX, MAX_SIZE_Y_PX, PIXEL_PITCH, optimalSieveStruct)

%parse the number of zones and number of pinholes.
N = optimalSieveStruct.N;
N_h = optimalSieveStruct.N_h;

%calculate space domain grid.
span_x = (MAX_SIZE_X_PX - 1)*PIXEL_PITCH;
xGrid = linspace(-span_x/2, span_x/2, MAX_SIZE_X_PX);
xGrid = repmat(xGrid, [MAX_SIZE_Y_PX, 1]);
span_y = (MAX_SIZE_Y_PX - 1)*PIXEL_PITCH;
yGrid = linspace(-span_y/2, span_y/2, MAX_SIZE_Y_PX);
yGrid = repmat(yGrid', [1, MAX_SIZE_X_PX]);

%Sample sieve to get discretized aperture function.
a = zeros(MAX_SIZE_Y_PX, MAX_SIZE_X_PX);
for cZones = 1:N % Iterates zones.
    for cHoles = 1:N_h(cZones) % Iterates pinholes.
        currentPinholeData = optimalSieveStruct.sieveData{1}{cZones}(:, cHoles);% b/c they are in mms.
        % calculate A_n for this pinhole.
        a_n = circ(xGrid,...
            yGrid, currentPinholeData(1),...
            currentPinholeData(2), currentPinholeData(3));
        a = a | a_n;
    end
end

sampledSieveStruct.a = a;
sampledSieveStruct.xGrid = xGrid;
sampledSieveStruct.yGrid = yGrid;
sampledSieveStruct.MAX_SIZE_X_PX = MAX_SIZE_X_PX;
sampledSieveStruct.MAX_SIZE_Y_PX = MAX_SIZE_Y_PX;
sampledSieveStruct.PIXEL_PITCH = PIXEL_PITCH;

end

