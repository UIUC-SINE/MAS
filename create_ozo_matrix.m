function [ozo_matrix] = create_ozo_matrix(ozo_vector,aa,p)

ozo_matrix = [];
for ii = 1:p
    temp = ozo_vector(ii*aa*aa-aa*aa+1:ii*aa*aa);
    ozo_matrix = [ozo_matrix;reshape(temp,aa,aa)];
end
