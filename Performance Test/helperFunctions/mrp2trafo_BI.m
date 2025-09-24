
function [T_BI] = mrp2trafo_BI(sigma)

T_BI = eye(3) + (8*cpm(sigma)^2 - 4*(1-sigma'*sigma)*cpm(sigma)) / ((1+sigma'*sigma)^2);

end