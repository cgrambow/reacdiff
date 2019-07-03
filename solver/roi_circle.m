function roi = roi_circle(xx,yy,center,radius,d,operation)
  %a diffuse shape function for union and intersection of circles, whose centers (N*2) and radii (N*1 or 1*N) are listed as arrays.
  %xx and yy can be 1D array in different directions or matrices of the same size
  %d is the diffuse layer thickness, it can be a scalar or two element for x and y directions
  %operation takes the roi of each circle as input and operate on it
  %if operation is not provided, we take the union (sum)
  N = size(center,1);
  if isscalar(d)
    d = d*ones(1,N);
  end
  union =  (nargin < 6) || isempty(operation);

  for i = 1:N
    r = sqrt((xx-center(i,1)).^2 + (yy-center(i,2)).^2);
    roi_i = 1./(1 + exp((r - radius(i))/d(i)));
    if union
      if i == 1
        roi = roi_i;
      else
        roi = max((roi + roi_i),1); %union operator
      end
    else
      roi_out{i} = roi_i;
    end
  end

  if ~union
    roi = feval(operation,roi_out{:});
  end
end
