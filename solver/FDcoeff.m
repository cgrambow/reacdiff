function y = FDcoeff(npts,h,deriv,x)
  %coefficients of 1D finite difference
  %ntps: number of points, h: distance
  %deriv: coefficients of the derivatives. It can be a 1D array, or if 2D, each column is a set of coefficient
  %in increasing order 0 through npts-1
  %for example, to get 4th order derivative, deriv = [0;0;0;0;1]
  %to get 2nd and 4th, deriv = [0,0;0,0;1,0;0,0;0,1];
  %if x is provided, x is the positions of the points, discarding npts and h information
  if nargin < 4 || isempty(x)
    n = (npts-1)/2;
    x = h*(-n:n);
  elseif size(x,2)==1
    x = x';
    npts = size(x,2);
  end
  if size(deriv,1)==1
    deriv = deriv';
  end
  if size(deriv,1)<npts
    deriv(npts,:) = 0;
  end
  deriv = deriv .* factorial(0:npts-1)';
  x = repmat(x,npts,1);
  x(1,:) = 1;
  x = cumprod(x,1);
  y = x\deriv;
end
