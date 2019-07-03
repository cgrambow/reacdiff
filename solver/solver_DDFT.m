function solver_DDFT()
addpath('../../CHACR/odesolver')
%in order to be consistent with formk, the first dimension is x and the second dimension is y
N = [128,128]*2;
L = [5,5];
dx = L./N;
n = prod(N);
n0 = 0.1;
sigma = 0.01;
rng(1);
y0 = n0 + sigma*randn(N);
y0 = y0(:);
[k2,k] = formk(N,L);
k0 = 10;
alpha = 2;
C = exp(-(sqrt(k2)-k0).^2/(2*alpha^2))*0.95;
tspan = linspace(0,4,100);
% tspan = [0,2];
t0 = tspan(1);

%Finite differencing operator in real space
for i = 1:2
  NI = N(3-i);
  ND = N(i);
  I = speye(NI);
  E = sparse(2:ND,1:ND-1,1,ND,ND);
  D = E+E'-2*I;
  %circulant D, periodic boundary condition
  D(1,ND) = 1;
  D(ND,1) = 1;
  if i==1
    L = kron(I,D)/dx(i)^2;
  else
    L = L + kron(D,I)/dx(i)^2;
  end
end
%Finite differencing operator in Fourier space
Lconv = [zeros(1,3); 1,-2,1; zeros(1,3)]/dx(2)^2 + [zeros(3,1), [1;-2;1], zeros(3,1)]/dx(1)^2;
LK = psf2otf(Lconv,N);

yp0 = RHS(t0,y0,N,dx,C);
options = odeset('InitialSlope',yp0,'Jacobian',@jacobian);
moreoptions = moreodeset('skipInit',true,'Krylov',true, ...
'pencil',@(xi,t,y,hinvGak,info) pencil(N,dx,C,xi,t,y,hinvGak,info), ...
'KrylovDecomp',@(~,~,dfdy,hinvGak) KrylovDecomp(L,dfdy,hinvGak), ...
'KrylovPrecon',@(x,L,U,hinvGak,~,~,~) KrylovPrecon(LK,N,C,x,L,U,hinvGak),...
'jacMult',@(xi,t,y,info) jacobian_mult(N,dx,C,xi,t,y,info),...
'gmrestol',1e-3);

[t,y] = myode15s(@(t,y) RHS(t,y,N,dx,C),tspan,y0,options,moreoptions);

end

function dy = RHS(t,y,N,dx,C)
  %C should be 2D, same below
  y = reshape(y,N);
%   mu = log(1+y);
  mu = y - y.^2/2 + y.^3/3;
  mu = mu - ifftn(C .* fftn(y));
  dy = 0;
  for i = 1:length(N)
    dy = dy + (circshift(mu,1,i)+circshift(mu,-1,i)-2*mu)/dx(i)^2;
  end
  dy = dy(:);
end

function dfdy = jacobian(t,y)
  %this only computes the nonlinear part, as an input to KrylovDecomp
%   dfdy = 1./(1+y);
  dfdy = 1 - y + y.^2;
end

function Pargs = KrylovDecomp(L,dfdy,hinvGak)
  %here dfdy comes from Jacobian
  %L is the finite-difference Laplacian operator
  n = length(L);
  J = speye(n) - hinvGak*L.*sparse(1:n,1:n,dfdy);
  [L,U] = ilu(J);
  Pargs = {L,U};
end

function yy = KrylovPrecon(LK,N,C,x,L,U,hinvGak)
  %LK is the Fourier transform of the finite difference Laplacian operator
  x = reshape(x,N);
  yy = ifft( fft(x) ./ (1 + hinvGak*LK.*C) );
  yy = yy(:);
  yy = U \ (L \ yy);
end

function yy = jacobian_mult(N,dx,C,xi,t,y,info)
  if isequal(info,'force')
    yy = jacobian_mult(N,dx,C,xi,t,y,[]);
    yy = jacobian_mult(N,dx,C,xi,t,y,yy);
  elseif isempty(info)
%     yy = 1./(1+y);
    yy = 1 - y + y.^2;
    yy = reshape(yy,N);
  else
    xi = reshape(xi,N);
    dfdy = info;
    mu = dfdy .* xi;
    mu = mu - ifftn(C .* fftn(xi));
    yy = 0;
    for i = 1:length(N)
      yy = yy + (circshift(mu,1,i)+circshift(mu,-1,i)-2*mu)/dx(i)^2;
    end
    yy = yy(:);
  end
end

function yy = pencil(N,dx,C,xi,t,y,hinvGak,info)
  yy = jacobian_mult(N,dx,C,xi,t,y,info);
  if ~isempty(info)
    yy = xi - hinvGak*yy;
  end
end
