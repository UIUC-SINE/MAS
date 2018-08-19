function [retVal] = circ(x, y, x_n, y_n, d_n)

retVal = (x - x_n).^2 + (y - y_n).^2 <= (d_n/2)^2;

end

