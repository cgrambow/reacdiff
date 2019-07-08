function varargout = solver_DDFT(tspan,y0,params,meta,mode,varargin)
%this provides solution to both forward and backward evaluation
%set mode = 'forward' (default) to perform model evaluation, in this mode, we accept two varargin, varargin{1} is sol, when true (false by default), output sol structure (varargin = {tout,sol,params},, when false, varargin = {tout,y,params}. varargin{2} is FSA, when true (false by default), also computes FSA and varargin{4} = ys.
%set mode = 'adjoint' to perform ASA analysis. varargin{1} = error, varargin{2] = discrete, tspan and y0 must be tdata and sol (solution history). varargout = {grad}
ASA = (nargin > 4) && isequal(mode,'adjoint');
if ASA
  error = varargin{1};
  discrete = varargin{2};
else
  sol = ~isempty(varargin) && varargin{1};
  FSA = length(varargin)>1 && varargin{2};
end

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

%the Jacobian and mass matrix of the system is Hermitian, so the treatment of pencil is the same for forward and ASA eval
moreoptions = moreodeset('skipInit',true,'Krylov',true, ...
'gmrestol',1e-4,'restart',10);

if ~ASA
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
  yp0 = RHS(t0,y0,params);
  options = odeset('InitialSlope',yp0,'Jacobian',@(t,y) jacobian(t,y,params));
  moreoptions = moreodeset(moreoptions, ...
  'jacMult',@(xi,t,y,info) jacobian_mult(params,xi,t,y,info), ...
  'pencil',@(xi,t,y,hinvGak,info) pencil(params,xi,t,y,hinvGak,info), ...
  'KrylovDecomp',@(~,~,dfdy,hinvGak) KrylovDecomp(L,dfdy,hinvGak), ...
  'KrylovPrecon',@(x,L,U,hinvGak,~,~,~) KrylovPrecon(LK,params,x,L,U,hinvGak));
  odeFcn = @(t,y) RHS(t,y,params);

  if FSA
    moreoptions = moreodeset(moreoptions, ...
    'FSA', true, ...
    'sensFcn', @(t,y) sensFcn(t,y,meta,params), ...
    'ys0', zeros(length(y0),meta.extdata.numParams), ...
    'ysp0', sensFcn(tspan(1),y0,meta,params));
  end

  if sol
    y = myode15s(odeFcn,tspan,y0,options,moreoptions);
    tout = y.x;
    ys = [];
  else
    [tout,y,ys] = myode15s(odeFcn,tspan,y0,options,moreoptions);
  end
  varargout = {tout,y,params,ys};
else
  addpath('../../CHACR/GIP')
  tdata = tspan;
  sol = y0;
  ASAQuadNp = 1;

  options = odeset('Jacobian',@(t,y) jacobian(t,sol,params), ...
  'mass', -speye(prod(params.N)),'MassSingular','yes','MStateDependence','none');
  moreoptions = moreodeset(moreoptions, ...
  'jacMult',@(xi,t,y,info) jacobian_mult(params,xi,t,sol,info), ...
  'pencil',@(xi,t,y,hinvGak,info) pencil(params,xi,t,sol,hinvGak,info,true), ...
  'KrylovDecomp',@(~,~,dfdy,hinvGak) KrylovDecomp(L,dfdy,hinvGak,true), ...
  'KrylovPrecon',@(x,L,U,hinvGak,~,~,~) KrylovPrecon(LK,params,x,L,U,hinvGak,true));
  if isfield(params,'Csensval')
    sensFcnMultiplier = false;
    sF = @sensFcn;
  else
    sensFcnMultiplier = true;
    sF = @sensFcn_ASA;
  end
  moreoptions = moreodeset(moreoptions, ...
  'interpFcn',@(flag,info,tnew,ynew,h,dif,k,idxNonNegative) ASA_gradient(flag,info,tnew,ynew,h,dif,k,idxNonNegative,sol,sF,{params},meta,ASAQuadNp,sensFcnMultiplier));

  if ~discrete
    tspan = [tdata(end),tdata(1)];
    F0 = error(end,:)';
    [y0,yp0] = ASAinit(F0,discrete,params,tspan(1),sol);
    options = odeset(options,'InitialSlope',yp0);
    moreoptions.linearMult = @(xi,t,info) ASA_mult(t,xi,sol,info,params,tdata,error);
    odeFcn = @(t,y) ASA_eqn(t,y,sol,params,tdata,error);
    grad = myode15s(odeFcn,tspan,y0,options,moreoptions);
  else
    y_final = 0;
    yp_final = 0;
    moreoptions.linearMult = @(xi,t,info) ASA_mult(t,xi,sol,info,params);
    odeFcn = @(t,y) ASA_eqn(t,y,sol,params);
    %initialize at each time step
    y0_list = zeros(n,length(tdata)-1);
    yp0_list = y0_list;
    for ind = 2:length(tdata)
      t0 = tdata(ind);
      F0 = error(end,:)';
      [y0_list(:,ind-1),yp0_list(:,ind-1)] = ASAinit(F0,discrete,params,t0,sol);
    end
    %serial
    grad = zeros(1,meta.extdata.numParams);
    for ind = (length(tdata)-1):-1:1
      tspan = [tdata(ind+1),tdata(ind)];
      y0 = y0_list(:,ind)+y_final;
      yp0 = yp0_list(:,ind)+yp_final;
      options = odeset(options,'InitialSlope',yp0);
      [grad_new,y_final,yp_final] = myode15s(odeFcn,tspan,y0,options,moreoptions);
      grad = grad + grad_new;
      if isnan(y_final)
        break;
      end
    end
  end
  varargout = {grad};
end

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
  %accommondate its usage through ASA when y is a struct
  if isstruct(y)
    y = sol_interp(y,t);
  end
  %this only computes the nonlinear part, as an input to KrylovDecomp
  dfdy = customizeFunGrad(params,'mu','grad',y);
end

function Pargs = KrylovDecomp(L,dfdy,hinvGak,adjoint)
  %here dfdy comes from Jacobian
  %L is the finite-difference Laplacian operator
  if nargin>3 && adjoint
    msign = -1;
  else
    msign = 1;
  end
  n = length(L);
  J = msign*speye(n) - hinvGak*L.*sparse(1:n,1:n,dfdy);
  [L,U] = ilu(J);
  Pargs = {L,U};
end

function yy = KrylovPrecon(LK,params,x,L,U,hinvGak,adjoint)
  %LK is the Fourier transform of the finite difference Laplacian operator
  if nargin>6 && adjoint
    msign = -1;
  else
    msign = 1;
  end
  N = params.N;
  C = params.C;
  x = reshape(x,N);
  yy = ifftn( fftn(x) ./ (msign + hinvGak*LK.*C) );
  yy = yy(:);
  yy = U \ (L \ yy);
end

function yy = jacobian_mult(params,xi,t,y,info)
  if isequal(info,'force')
    yy = jacobian_mult(params,xi,t,y,[]);
    yy = jacobian_mult(params,xi,t,y,yy);
  elseif isempty(info)
    if isstruct(y)
      y = sol_interp(y,t);
    end
    yy = customizeFunGrad(params,'mu','grad',y);
    yy = reshape(yy,params.N);
  else
    N = params.N;
    dx = params.dx;
    C = params.C;
    yy = zeros(size(xi));
    for p = 1:length(size(xi,2))
      xip = reshape(xi(:,p),N);
      dfdy = info;
      mu = dfdy .* xip;
      mu = mu - ifftn(C .* fftn(xip));
      for i = 1:length(N)
        yy(:,p) = yy(:,p) + reshape((circshift(mu,1,i)+circshift(mu,-1,i)-2*mu)/dx(i)^2, [], 1);
      end
    end
  end
end

function yy = ASA_mult(t,xi,sol,info,params,varargin)
  if isempty(info)
    yy = jacobian_mult(params,[],t,sol,[]);
  else
    yy = jacobian_mult(params,xi,[],[],info);
    if ~isempty(varargin)
      tdata = varargin{1};
      error = varargin{2};
      [interval,alpha] = interval_counter(tdata,t,length(tdata)-1,2);
      source = (error(interval,:)*(1-alpha) + error(interval+1,:)*alpha)';
      yy = yy + source;
    end
  end
end

function dy = ASA_eqn(t,y,sol,params,varargin)
  %wrapper function for odeFcn (not really needed, mostly just a placeholder, except for daeic)
  info = ASA_mult(t,y,sol,[],params);
  dy = ASA_mult(t,y,sol,info,params,varargin{:});
end

function yy = pencil(params,xi,t,y,hinvGak,info,adjoint)
  if nargin>6 && adjoint
    msign = -1;
  else
    msign = 1;
  end
  yy = jacobian_mult(params,xi,t,y,info);
  if ~isempty(info)
    yy = msign*xi - hinvGak*yy;
  end
end

function dy = sensFcn(t,y,meta,params)
  N = params.N;
  dx = params.dx;
  y = reshape(y,N);
  names = fieldnames(meta);
  numParams = meta.extdata.numParams;
  dy = zeros(numel(y),numParams);

  for i = 1:numel(names)
    name = names{i};
    if isequal(name,'extdata')
      continue
    end
    paramsIndex = meta.(name).index;
    switch name
    case 'mu'
      mu = customizeSensEval(params,name,y);
    case 'C'
      if isfield(params,'Csensval')
        mu = - ifft2(params.Csensval .* fftn(y),'symmetric');
      else
        mu = -params.Csens(y);
      end
    end
    dyi = 0;
    for j = 1:length(N)
      dyi = dyi + (circshift(mu,1,j)+circshift(mu,-1,j)-2*mu)/dx(j)^2;
    end
    dy(:,paramsIndex) = reshape(dyi,[],numel(paramsIndex));
  end
end

function dy = sensFcn_ASA(t,y,omega,meta,params)
  N = params.N;
  dx = params.dx;
  y = reshape(y,N);
  names = fieldnames(meta);
  numParams = meta.extdata.numParams;
  dy = zeros(1,numParams);

  for i = 1:numel(names)
    name = names{i};
    if isequal(name,'extdata')
      continue
    end
    paramsIndex = meta.(name).index;
    switch name
    case 'mu'
      mu = customizeSensEval(params,name,y);
      dyi = 0;
      for j = 1:length(N)
        dyi = dyi + (circshift(mu,1,j)+circshift(mu,-1,j)-2*mu)/dx(j)^2;
      end
      dy(paramsIndex) = sum(omega.*dyi(:));
    case 'C'
      for p = 1:length(paramsIndex)
        mu = -params.Csens(y,p);
        dyi = 0;
        for j = 1:length(N)
          dyi = dyi + (circshift(mu,1,j)+circshift(mu,-1,j)-2*mu)/dx(j)^2;
        end
        dy(paramsIndex(p)) = sum(omega.*dyi(:));
      end
    end
  end
end

function [y0,yp0] = ASAinit(F0,discrete,params,t0,sol)
  y0 = zeros(size(F0));
  yp0 = F0;
  if discrete
    y0 = yp0;
    yp0 = jacobian_mult(params,y0,t0,sol,'force');
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
