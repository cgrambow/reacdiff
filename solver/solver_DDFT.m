function [t,y,params] = solver_DDFT(tspan,y0,params)
addpath('../../CHACR/odesolver')
%in order to be consistent with formk, the first dimension is x and the second dimension is y
if ~isfield(params,'dx') && isfield(params,'N') && isfield(params,'L')
  params.dx = params.L ./ params.N;
elseif ~isfield(params,'N') && isfield(params,'dx') && isfield(params,'L')
  params.N = params.L ./ params.dx;
elseif ~isfield(params,'L') && isfield(params,'N') && isfield(params,'dx')
  params.L = params.N .* params.dx;
elseif ~all(isfield(params,{'dx','N','L'}))
  params.N = [128,128];
  params.L = [5,5];
  params.dx = params.L./params.N;
end
n = prod(params.N);
N = params.N;
L = params.L;
dx = params.dx;
if ~isfield(params,'C')
  %default
  [k2,k] = formk(N,L);
  k0 = 10;
  alpha = 2;
  params.C = exp(-(sqrt(k2)-k0).^2/(2*alpha^2))*0.95;
end
if ~isfield(params,'mu')
  params.mu.func = @(x,coeff) x - x.^2/2 + x.^3/3; %@(x) log(1+x)
  params.mu.grad = @(x,coeff) 1 - x + x.^2;
  params.mu.params = [];
end
% mu.func = @(x,coeff) log((1+x)./(1-x))/2;
% mu.grad = @(x,coeff) 1./(1-x.^2);
% mu.params = [];
if isempty(y0) || isscalar(y0)
  %initialization
  if isempty(y0)
    n0 = 0.1;
  else
    n0 = y0;
  end
  sigma = 0.01;
  rng(1);
  y0 = n0 + sigma*randn(N);
  y0 = y0(:);
end
if isempty(tspan)
  tspan = linspace(0,4,100);
end
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

yp0 = RHS(t0,y0,params);
options = odeset('InitialSlope',yp0,'Jacobian',@(t,y) jacobian(t,y,params));
moreoptions = moreodeset('skipInit',true,'Krylov',true, ...
'pencil',@(xi,t,y,hinvGak,info) pencil(params,xi,t,y,hinvGak,info), ...
'KrylovDecomp',@(~,~,dfdy,hinvGak) KrylovDecomp(L,dfdy,hinvGak), ...
'KrylovPrecon',@(x,L,U,hinvGak,~,~,~) KrylovPrecon(LK,params,x,L,U,hinvGak),...
'jacMult',@(xi,t,y,info) jacobian_mult(params,xi,t,y,info),...
'gmrestol',1e-3);

[t,y] = myode15s(@(t,y) RHS(t,y,params),tspan,y0,options,moreoptions);

end

function dy = RHS(t,y,params)
  %C should be 2D, same below
  N = params.N;
  dx = params.dx;
  C = params.C;
  y = reshape(y,N);
  mu = customizeFunGrad(params,'mu','fun',y);
  mu = mu - ifftn(C .* fftn(y));
  dy = 0;
  for i = 1:length(N)
    dy = dy + (circshift(mu,1,i)+circshift(mu,-1,i)-2*mu)/dx(i)^2;
  end
  dy = dy(:);
end

function dfdy = jacobian(t,y,params)
  %this only computes the nonlinear part, as an input to KrylovDecomp
  dfdy = customizeFunGrad(params,'mu','grad',y);
end

function Pargs = KrylovDecomp(L,dfdy,hinvGak)
  %here dfdy comes from Jacobian
  %L is the finite-difference Laplacian operator
  n = length(L);
  J = speye(n) - hinvGak*L.*sparse(1:n,1:n,dfdy);
  [L,U] = ilu(J);
  Pargs = {L,U};
end

function yy = KrylovPrecon(LK,params,x,L,U,hinvGak)
  %LK is the Fourier transform of the finite difference Laplacian operator
  N = params.N;
  C = params.C;
  x = reshape(x,N);
  yy = ifft( fft(x) ./ (1 + hinvGak*LK.*C) );
  yy = yy(:);
  yy = U \ (L \ yy);
end

function yy = jacobian_mult(params,xi,t,y,info)
  if isequal(info,'force')
    yy = jacobian_mult(params,xi,t,y,[]);
    yy = jacobian_mult(params,xi,t,y,yy);
  elseif isempty(info)
    yy = customizeFunGrad(params,'mu','grad',y);
    yy = reshape(yy,params.N);
  else
    N = params.N;
    dx = params.dx;
    C = params.C;
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

function yy = pencil(params,xi,t,y,hinvGak,info)
  yy = jacobian_mult(params,xi,t,y,info);
  if ~isempty(info)
    yy = xi - hinvGak*yy;
  end
end

function varargout = customizeFunGrad(customize,name,request,varargin)
  %allow func to output both fun and grad eval
  %request can be 'fun','grad',or 'fungrad', and the output will be the requested evaluation in the corresponding order.
  %evaluate fun and grad from func function at the same time if grad doesn't exist
  funstr = customize.(name);
  params = funstr.params;
  fungrad = (~isfield(funstr,'grad'));
  if fungrad
    [fun,grad] = funstr.func(varargin{:},params);
  else
    if isequal(request,'fun') || isequal(request,'fungrad')
      fun = funstr.func(varargin{:},params);
    end
    if isequal(request,'grad') || isequal(request,'fungrad')
      grad = funstr.grad(varargin{:},params);
    end
  end
  switch request
  case 'fun'
    varargout = {fun};
  case 'grad'
    varargout = {grad};
  case 'fungrad'
    varargout = {fun,grad};
  end
end
