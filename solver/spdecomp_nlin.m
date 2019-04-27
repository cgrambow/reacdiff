function f = spdecomp_nlin(t,y,N,k2,omega)
  y = reshape(y,N);
  yy = real(ifftn(y));
  mu = log(yy./(1-yy))+omega*(1-2*yy);
  f = fftn(mu);
  f = -f(:).*k2;
end
