function dy = DDFT_nlin(t,y,k2,N,y00)
  if nargin > 4 && ~isempty(y00)
    %k=0 component is omitted
    k0flag = true;
    y = [y00;y(:)];
  else
    k0flag = false;
  end
  y = real(ifftn(reshape(y,N)));
  % mu = y - y.^2/2 + y.^3/3;
  mu = log(1+y);
  fmu = reshape(fftn(mu,N),[],1);
  if k0flag
    fmu(1) = [];
  end
  dy = -k2 .* fmu;
end
