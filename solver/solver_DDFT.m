function solver_DDFT()
%in order to be consistent with formk, the first dimension is x and the second dimension is y

%Finite differencing operator in real space
I = speye(n,n);
E = sparse(2:n,1:n-1,1,n,n);
D = E+E'-2*I;
%circulant D, periodic boundary condition
D(1,n) = 1;
D(n,1) = 1;
L = kron(D,I)+kron(I,D);
%Finite differencing operator in Fourier space
Lconv = [zeros(1,3); 1,-2,1; zeros(1,3)]/dx(2)^2 + [zeros(3,1), [1;-2;1], zeros(3,1)]/dx(1)^2;
LK = psf2otf(Lconv,N);

end

function dy = RHS(t,y,N,dx,C)
  %C should be 2D, same below
  y = reshape(y,N);
  mu = log(1+y);
  mu = mu - ifftn(C .* fftn(y));
  dy = 0;
  for i = 1:length(N)
    dy = dy + (circshift(mu,1,i)+circshift(mu,-1,i)-2*mu)/dx(i)^2;
  end
end

function dfdy = jacobian(t,y)
  %this only computes the nonlinear part, as an input to KrylovDecomp
  dfdy = 1./(1-y);

end

function Pargs = KrylovDecomp(L,Mt,dMpsidy,dfdy,hinvGak)
  %here dfdy comes from Jacobian
  %L is the finite-difference Laplacian operator
  n = length(L);
  J = speye(n) - hinvGak*L.*sparse(1:n,1:n,dfdy);
  [L,U] = ilu(J);
  Pargs = {L,U};
end

function yy = KrylovPrecon(LK,N,C,x,L,U,hinvGak,~,~,~)
  %LK is the Fourier transform of the finite difference Laplacian operator
  x = reshape(x,N);
  yy = ifft( fft(x) ./ (1 + hinvGak*LK.*C) );
  yy = U \ (L \ yy);
end

function yy = pencil(N,C,xi,t,y,hinvGak,info)
  if isempty(info)
    yy = 1./(1-y);
  else
    xi = reshape(xi,N);
    dfdy = info;
    mu = dfdy .* xi;
    mu = mu - ifftn(C .* fftn(y));
    yy = 0;
    for i = 1:length(N)
      yy = dy + (circshift(mu,1,i)+circshift(mu,-1,i)-2*mu)/dx(i)^2;
    end
  end
end
