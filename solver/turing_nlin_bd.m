function f = turing_nlin_bd(t,y,A,b,lb,ub,N)
  %constant term b
  %A matrix for linear reaction terms (nspecies-by-nspecies)
  %lb and ub are lower and upper boundary for the reaction rate for each species
  %N is the size of the grid
  nspecies = length(b);
  y = reshape(y,prod(N),nspecies);
  for i=1:nspecies
    y(:,i) = reshape(real(ifftn(reshape(y(:,i),N))),[],1);
  end
  if size(b,2)==1
    b = b';
  end
  f = y*A' + b;
  if size(lb,2)==1
    lb = lb';
  end
  if size(ub,2)==1
    ub = ub';
  end
  f = min(max(f,lb),ub);
  for i=1:nspecies
    f(:,i) = reshape(fftn(reshape(f(:,i),N)),[],1);
  end
  f = f(:);
end
