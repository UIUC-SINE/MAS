function [g_squared, xGrid, yGrid] = calculate2DPSF(optimalSieveStruct, DOF_span, HORIZONTAL_SAMPLING_CNT, VERTICAL_SAMPLING_CNT, SAMPLING_INTERVAL, out)

p = Inf; % Separation of source and photon sieve.
NUMBER_OF_ZONES_TO_BE_PROCESSED = 9999; % Maximum number of zones of sieve to be processed. 

%% Calculate minimum fabricable structure for given sieve.
effectiveZoneCountOfSieve = min(length(optimalSieveStruct.sieveData{1}), NUMBER_OF_ZONES_TO_BE_PROCESSED);
MFS = optimalSieveStruct.sieveData{1, end}{effectiveZoneCountOfSieve}(3, 1); %effective minimum fabricable structure.
DOF = 2.*(MFS.^2)./optimalSieveStruct.TARGET_WL; % effective DOF of focus of sieve.


%% Calculate corresponding d_i values. Are set according to the requested
% DOF span and optimal focus.
d_i = double(DOF_span).*(DOF) + optimalSieveStruct.f; % corresponding d_i values. From DOF_span

%% Initialize sampling points in tinme domain, in meters.
xGrid = (-(HORIZONTAL_SAMPLING_CNT-1)/2:1:(HORIZONTAL_SAMPLING_CNT-1)/2).*SAMPLING_INTERVAL;
xGrid = repmat(xGrid, VERTICAL_SAMPLING_CNT, 1);
yGrid = (-(VERTICAL_SAMPLING_CNT-1)/2:1:(VERTICAL_SAMPLING_CNT-1)/2)'.*SAMPLING_INTERVAL;
yGrid = repmat(yGrid, 1, HORIZONTAL_SAMPLING_CNT);

%% Init sampling points on frequency domain.
delta_t_x = SAMPLING_INTERVAL;
delta_t_y = SAMPLING_INTERVAL;

span_f_x = 1/delta_t_x;
span_f_y = 1/delta_t_y;

x_vals_freq = linspace(-span_f_x/2, span_f_x/2, HORIZONTAL_SAMPLING_CNT);
x_vals_freq = repmat(x_vals_freq, VERTICAL_SAMPLING_CNT, 1);
y_vals_freq = linspace(-span_f_y/2, span_f_y/2, VERTICAL_SAMPLING_CNT)';
y_vals_freq = repmat(y_vals_freq, 1, HORIZONTAL_SAMPLING_CNT);


%% Generate A function for all sieves.
a = cell(1, length(DOF_span)); % Var to hold PSFs.
%% PSF calculation. Based on the paper 'IMAGE FORMATION MODEL FOR PHOTON SIEVES' by Oktem et. al.
g_normalized = cell(1, length(DOF_span)); % Container of normalized PSFs.
g_squared = cell(1, length(DOF_span)); % Container of PSFs.
    for flcounter = 1:length(DOF_span)
        printf('iterations %d\n', flcounter)
        fflush(stdout);
        a{1, flcounter} = zeros(VERTICAL_SAMPLING_CNT, HORIZONTAL_SAMPLING_CNT);
        for zoneCounter = 1:min(length(optimalSieveStruct.sieveData{1}), NUMBER_OF_ZONES_TO_BE_PROCESSED) % Iterates zones.
            ## fprintf('PSF Calculation. CurrentZone: %d\n', zoneCounter);
            %Calculate current set of x and y values in the frequnecy
            %domain.
            current_x_vals_freq = -x_vals_freq.*(optimalSieveStruct.TARGET_WL).*d_i(1, flcounter);
            current_y_vals_freq = -y_vals_freq.*(optimalSieveStruct.TARGET_WL).*d_i(1, flcounter);
            
            for holeCounter = 1:size(optimalSieveStruct.sieveData{1}{zoneCounter}, 2) % Iterates pinholes.
                currentPinholeData = optimalSieveStruct.sieveData{1}{zoneCounter}(:, holeCounter);
                % calculate A_n for this pinhole.
                a_n = circ(current_x_vals_freq,...
                    current_y_vals_freq, currentPinholeData(1),...
                    currentPinholeData(2), currentPinholeData(3)); 
                a{1, flcounter} = a{1, flcounter} | a_n; % Aperture function
            end
        end
        
        flc = d_i(1, flcounter); % Current d_i value. Separation between aperture and measurement planes.
        delta = 1/flc + 1/p; % Corresponding delta in paper.
        currentExponent = exp((1i.*pi.*delta.*optimalSieveStruct.TARGET_WL.*(flc.^2)).*(x_vals_freq.^2 + y_vals_freq.^2));
        current_g = fftshift(ifft2(currentExponent.*a{1, flcounter})); % Convolution in Fourier domain.
        g_squared{1, flcounter} = abs(current_g).^2;
        g_normalized{1, flcounter} = g_squared{1, flcounter}./max(max(g_squared{1, flcounter}));
    end
    
    clear x_vals_freq y_vals_freq;
    
    for flcounter = 1:length(DOF_span)
        %Writes the normalized PSFs to output folder. Max value of each set
        %to 255.
        imwrite(uint8(255.*g_normalized{1, flcounter}), [out '/2D_NORM_PSF_DOF_' num2str(DOF_span(flcounter)) '.png']);
    end
end



