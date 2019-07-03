function roi = roi_rectangle(xx,yy,xrange,yrange,d)
  %a diffuse shape function for a rectangle, x > xrange(1), x < xrange(2), y > yrange(1), y < yrange(2)
  %use Inf or NaN to indicate no bound
  %xx and yy are arrays of the same size
  %d is the diffuse layer thickness, it can be a scalar or two element for x and y directions
  if isscalar(d)
    d = [d,d];
  end
  roi = ones(size(xx));
  for i = 1:2
    center = xrange(i);
    if isfinite(center)
      s = 2*i-3;
      roi = roi.*(1./(1+exp(s*(xx-center)/d(1))));
    end
  end
  for i = 1:2
    center = yrange(i);
    if isfinite(center)
      s = 2*i-3;
      roi = roi.*(1./(1+exp(s*(yy-center)/d(2))));
    end
  end
