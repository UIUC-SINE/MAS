function [ozo_vector] = create_ozo_vector(ozo_matrix,aa,p)

ozo_vector = [];
for ii = 1:p
    temp = ozo_matrix(ii*aa-aa+1:ii*aa,:);
    ozo_vector = [ozo_vector;temp(:)];
end