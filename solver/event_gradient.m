function y = event_gradient(t,y,k2,thresh,component)
  %when the L2 norm of the gradient field is below thresh, the returned y is true, otherwise false.
  %note that y is in Fourier space.
  %L2 norm of the gradient field is \int_{|| \nabla y(x) ||^2 dx} = \int_{|| k ||^2 ||y(k)||^2 dk}
  %k2 should correspond to each row of y
  %component (1 by default) is the index of the species
  if nargin < 5
    component = 1;
  end
  dy = gradient_norm(y,k2,component);
  y = (dy<thresh);
end
