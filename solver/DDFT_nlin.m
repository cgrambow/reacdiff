function dy = DDFT_nlin(t,y,k2,N)
  y = real(ifftn(reshape(y,N)));
  mu = log(1+y);
  dy = -k2 .* reshape(fftn(mu,N),[],1);
end
