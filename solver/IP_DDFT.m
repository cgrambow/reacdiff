function [x_opt,fval,exitflag,pp] = IP_DDFT(tdata,ydata,params,kernelSize,Cspace,options,x_guess,varargin)
ps = inputParser;
addParameter(ps,'mode','IP'); %IP, eval, pp (walks through the function without eval or IP)
%set eval = true to only do the forward solve and return
addParameter(ps,'tspan',100);
%set tspan to a positive integer or 'sol' to specify the number of time points returned for solution history. Inactive and automatically set to 'discrete' if discrete = true
addParameter(ps,'bound',[]); %this is more Cspace = isotropic_CmE
addParameter(ps,'Nmu',0,@(x) (mod(x,2)==1)); %number of parameters for mu, must be odd
addParameter(ps,'D',false); %setting this to true turns on optimizing over D
addParameter(ps,'discrete',false);
ps.CaseSensitive = false;
parse(ps,varargin{:});
tspan = ps.Results.tspan;
mode = ps.Results.mode;
Nmu = ps.Results.Nmu;
discrete = ps.Results.discrete;
if discrete
  tspan = 'discrete';
end

addpath('../../CHACR/GIP')

ybound = [min(ydata(:)),max(ydata(:))];

switch Cspace
case 'isotropic'
  NC = kernelSize;
  meta.C.exp = false(1,NC);
  meta.C.exp(end) = true;
  [k2,~] = formk(params.N,params.L);
  p = custom_Poly;
  if Nmu>0
    %throw the constant term to mu, note that kernelSize is now the number of non-constant polynomials
    Csensval = feval(p.sens,k2,ones(1,NC+1));
    Csensval(:,:,1) = [];
  else
    Csensval = feval(p.sens,k2,ones(1,NC));
  end
  %let the basis of the last derivative be negative
  Csensval(:,:,end) = -Csensval(:,:,end);
  params.Csensval = Csensval;
case 'isotropic_CmE'
  %constant minus exponential
  NC = kernelSize;
  meta.C.exp = false;
  [k2,~] = formk(params.N,params.L);
  if isempty(ps.Results.bound)
    bound = [0,max(k2(:))];
  else
    bound = ps.Results.bound;
  end
  expleg = custom_ExpLegendre(1,bound);
  if Nmu>0
    %minus exponential only
    params.Cfunc.func = @(coeff) -expleg.func(k2,coeff);
    params.Cfunc.sens = @(coeff) -expleg.sens(k2,coeff);
  else
    params.Cfunc.func = @(coeff) coeff(1)-expleg.func(k2,coeff(2:end));
    params.Cfunc.sens = @(coeff) CmE_sens(k2,coeff,expleg.sens);
  end
case 'FD'
  NC = kernelSize;
  meta.C.exp = false(1,NC);
  meta.C.exp(end) = true;
  params = FD2otf(NC,params);
otherwise
  NC = floor((prod(kernelSize)+1)/2);
  params.Csens = @(y,i) Csens_ASA(y,i,Cspace,kernelSize);
  meta.C.exp = false;
end

meta.C.index = 1:NC;
if nargin < 7 || isempty(x_guess)
  x_guess = zeros(1,NC);
end

numParams = NC;
if Nmu>0
  meta.mu.index = numParams+(1:Nmu);
  meta.mu.exp = false(1,Nmu);
  meta.mu.exp(end) = true; %the coefficient of the higher odd order term must be positive
  params.mu = ChemPotential_Legendre(1,ybound,false,'onlyEnthalpy');
  if length(x_guess)<(numParams+Nmu)
    mu_guess = zeros(1,Nmu);
    mu_guess(1) = 2;
    %the following makes mu nonmonotonic if Nmu>1
    % mu_guess = [zeros(1,Nmu-1),1];
    % [~,dmu] = feval(params.mu.func,ybound(2),mu_guess);
    % %set the gradient at ybound to be 2
    % mu_guess(end) = log(2/dmu);
    x_guess = [x_guess(1:numParams), mu_guess];
  end
  numParams = NC+Nmu;
end

if ps.Results.D
  meta.D.index = numParams+1;
  meta.D.exp = true;
  if length(x_guess)<(numParams+1)
    D_guess = -3;
    x_guess = [x_guess(1:numParams), D_guess];
  end
end

switch mode
case 'eval'
  [x_opt,~,params] = IP(tdata,ydata,x_guess,meta,params, ...
  @(tdata,y0,FSA,meta,params) forwardSolver(tdata,y0,FSA,meta,params,tspan,ybound), ...
  [], [], [],...
  @(name,xparam,params) assign(name,xparam,params,Cspace,kernelSize),'eval',true);
  fval = [];
  exitflag = [];
case 'IP'
  loss = @(y,ydata,~) MSE(y,ydata,prod(params.dx)*100);
  lossHess = @(dy,~,~,~) MSE([],[],prod(params.dx)*100,dy);
  if nargin > 5 && ~isempty(options)
    options = optimoptions(options,'SpecifyObjectiveGradient',true,'Display','iter-detailed');
  else
    options = optimoptions('fminunc','SpecifyObjectiveGradient',true,'Display','iter-detailed');
  end
  [x_opt,fval,exitflag] = fminunc(@(x) IP(tdata,ydata,x,meta,params, ...
  @(tdata,y0,FSA,meta,params) forwardSolver(tdata,y0,FSA,meta,params,tspan,ybound), ...
  @adjointSolver, ...
  loss,lossHess, ...
  @(name,xparam,params) assign(name,xparam,params,Cspace,kernelSize),'discrete',discrete), ...
  x_guess, options);
end

if nargout > 3
  %post processing
  if isequal(Cspace,'isotropic')
    if Nmu>0
      params.Cfunc.func = @(x,coeff) p.func(x.^2,[0,coeff(1:end-1),-coeff(end)]);
    else
      params.Cfunc.func = @(x,coeff) p.func(x.^2,[coeff(1:end-1),-coeff(end)]);
    end
  elseif isequal(Cspace,'isotropic_CmE')
    if Nmu>0
      params.Cfunc.func = @(x,coeff) -expleg.func(x.^2,coeff);
    else
      params.Cfunc.func = @(x,coeff) coeff(1)-expleg.func(x.^2,coeff(2:end));
    end
  end
  pp.params = params;
  pp.meta = meta;
end

end

function params = assign(name,xparam,params,Cspace,kernelSize)
  switch name
  case 'mu'
    params.(name).params = xparam;
  case 'D'
    params.(name) = xparam;
  case 'C'
    %xparam for C is cut in half due to symmetry.
    %For the purpose of symmetry, we require that the kernel size be odd in both dimensions
    %if a kernel size is even, it is not a symmetric convolution
    %If the size of the kernel provided is N(1)*N(2), then the size of xparam is n = floor((prod(N)+1)/2)
    %xparam correspond to the first n in the kernel in linear index,
    %the rest is obtained using symmetry. This applies to both real space and k space
    %Both real space and k space representation is centered at the kernel center (not 1,1)
    %but it must be converted to be centered at 1,1 in the k-space representation used in DDFT solver
    %consider other extensions:
    %1. Reduce to representing 1D C(k) if it's axisymmetric
    %Cspace = isotropic, C(k) = a1 + a2*k^2 + a3*k^4 + ...
    %Cspace = FD, C = a1 - a2*nabla + a3*nabla^2 + ...
    switch Cspace
    case {'isotropic','FD'}
      coeff(1,1,:) = xparam;
      params.C = sum(params.Csensval .* coeff,3);
      params.Cfunc.params = xparam;
    case 'isotropic_CmE'
      params.C = params.Cfunc.func(xparam);
      params.Csensval = params.Cfunc.sens(xparam);
      params.Cfunc.params = xparam;
    case {'k','real'}
      if any(mod(kernelSize,2)==0)
        error('kernel size must be odd');
      end
      n = length(xparam);
      if n ~= floor((prod(kernelSize)+1)/2)
        error('kernel size incompatible with C input');
      end
      %note that n is odd
      C = xparam(:);
      %symmetry condition
      C = [C; flip(C(1:end-1))];
      C = reshape(C, kernelSize);
      switch Cspace
      case 'k'
        % Circularly shift so that the "center" of the OTF is at the
        % (1,1) element of the array. (from psf2otf)
        padSize = params.N - kernelSize;
        C       = padarray(C, padSize, 'post');
        C       = circshift(C,-floor(kernelSize/2));
        params.C = C;
      case 'real'
        params.C = psf2otf(C, params.N);
      end
    end
  end
end

function [tout,y,dy,params] = forwardSolver(tdata,y0,FSA,meta,params,tspan,ybound)
  %tspan can either be sol or number of time points
  if ~isfield(params,'mu') || isempty(params.mu)
    dmu = 1 - ybound + ybound.^2;
  else
    dmu = customizeFunGrad(params,'mu','grad',ybound);
  end
  if any(dmu-max(params.C(:))<0)
    %instability even at the max or min y value
    tout = NaN;
    y = NaN;
    dy = NaN;
    return
  end
  switch tspan
  case 'sol'
    sol = true;
  case 'discrete'
    sol = false;
  otherwise
    sol = false;
    tdata = linspace(tdata(1),tdata(end),tspan);
  end
  [tout,y,params,dy] = solver_DDFT(tdata,y0,params,meta,'forward','sol',sol,'FSA',FSA);
end

function grad = adjointSolver(tdata,sol,lossgrad,meta,discrete,~,params)
  grad = solver_DDFT(tdata,sol,params,meta,'adjoint','error',lossgrad,'discrete',discrete);
end

function dy = Csens(y,Cspace,kernelSize)
  %Csens takes up a lot of memory!! Csens is now a function handle for multiplying the sensitivity of C with a vector y. (dy = dC * y) Note that y should in the image format and in real space.
  %the returned dy has the dimension of size(y,1)*size(y,2)*number of parameters
  %for example, if real space representation is used, use circshift, in sensFcn,
  %if k space representation is used, for C(k), the sensitivity of -mu is y(k)exp(ikx)+y(-k)exp(-ikx)
  %since y is real, this is equal to 2Re(y(k)exp(ikx))
  %again, note that kernelSize must be odd
  n = floor((prod(kernelSize)+1)/2);
  kernelCenter = ceil(kernelSize/2);
  dy = zeros([size(y),n]);
  if isequal(Cspace,'k')
    imageSize = size(y);
    numPixel = prod(imageSize);
    y = fftn(y);
    [p1,p2] = ndgrid(0:(imageSize(1)-1),0:(imageSize(2)-1));
    p1 = p1/imageSize(1);
    p2=  p2/imageSize(2);
  end
  for i = 1:n
    [ind(1),ind(2)] = ind2sub(kernelSize,i);
    ind = ind - kernelCenter;
    switch Cspace
    case 'k'
      if all(ind==0)
        dy(:,:,i) = y(1,1)/numPixel;
      else
        if ind(1)>0
          ind(2) = imageSize(2)+ind(2);
        else
          ind = -ind;
        end
        %manually implement ifft to be faster see doc for ifftn
        dy(:,:,i) = 2*real(y(1+ind(1),1+ind(2)) * exp(2*pi*1i*(p1*ind(1)+p2*ind(2)))/numPixel);
      end
      % %the following code is another version consistent with the definition of C in assign
      % C = zeros(1,n);
      % C(i) = 1;
      % C = [C,flip(C(1:end-1))];
      % C = reshape(C, kernelSize);
      % padSize = imageSize - kernelSize;
      % C       = padarray(C, padSize, 'post');
      % C       = circshift(C,-floor(kernelSize/2));
      % dy(:,:,i) = ifftn(C .* y);
    case 'real'
      if all(ind==0)
        dy(:,:,i) = y;
      else
        dy(:,:,i) = circshift(y,ind) + circshift(y,-ind);
      end
    end
  end
end

function dy = Csens_ASA(y,i,Cspace,kernelSize)
  %another version of Csens, output only the sensitivity with respect to the ith parameter
  %Csens takes up a lot of memory!! Csens is now a function handle for multiplying the sensitivity of C with a vector y. (dy = dC * y) Note that y should in the image format and in real space.
  %the returned dy has the dimension of size(y,1)*size(y,2)*number of parameters
  %for example, if real space representation is used, use circshift, in sensFcn,
  %if k space representation is used, for C(k), the sensitivity of -mu is y(k)exp(ikx)+y(-k)exp(-ikx)
  %since y is real, this is equal to 2Re(y(k)exp(ikx))
  %again, note that kernelSize must be odd
  n = floor((prod(kernelSize)+1)/2);
  kernelCenter = ceil(kernelSize/2);
  if isequal(Cspace,'k')
    imageSize = size(y);
    numPixel = prod(imageSize);
    y = fftn(y);
    [p1,p2] = ndgrid(0:(imageSize(1)-1),0:(imageSize(2)-1));
    p1 = p1/imageSize(1);
    p2=  p2/imageSize(2);
  end
  [ind(1),ind(2)] = ind2sub(kernelSize,i);
  ind = ind - kernelCenter;
  switch Cspace
  case 'k'
    if all(ind==0)
      dy = y(1,1)/numPixel;
    else
      if ind(1)>0
        ind(2) = imageSize(2)+ind(2);
      else
        ind = -ind;
      end
      %manually implement ifft to be faster see doc for ifftn
      dy = 2*real(y(1+ind(1),1+ind(2)) * exp(2*pi*1i*(p1*ind(1)+p2*ind(2)))/numPixel);
    end
    % %the following code is another version consistent with the definition of C in assign
    % C = zeros(1,n);
    % C(i) = 1;
    % C = [C,flip(C(1:end-1))];
    % C = reshape(C, kernelSize);
    % padSize = imageSize - kernelSize;
    % C       = padarray(C, padSize, 'post');
    % C       = circshift(C,-floor(kernelSize/2));
    % dy = ifftn(C .* y);
  case 'real'
    if all(ind==0)
      dy = y;
    else
      dy = circshift(y,ind) + circshift(y,-ind);
    end
  end
end

function params = FD2otf(numDeriv,params)
 %transform finite difference coefficient to otf
 %y(:,:,i) is the otf of the 2(i-1)th derivative
 %in total, there are numDeriv derivatives
 npts = numDeriv*2-1;
 %switch to 0, k^2, k^4, ... rather than 0, nabla^2, nabla^4
 %but, to be consistent with Cspace = 'isotropic'
 %let the basis of the last derivative be negative
 derivcoeff = ones(numDeriv,1);
 derivcoeff(2:2:end) = -1;
 derivcoeff(end) = -1;
 derivdiag = accumarray([1:2:npts;1:numDeriv]',derivcoeff);
 coeffx = FDcoeff(npts,params.dx(1),derivdiag);
 coeffy = FDcoeff(npts,params.dx(2),derivdiag);
 row = [numDeriv*ones(1,npts),1:npts];
 column = [1:npts,numDeriv*ones(1,npts)];
 C = zeros(params.N(1),params.N(2),numDeriv);
 for i = 1:numDeriv
   if i==1
     kernel = 1;
   else
     kernel = accumarray([row',column'],[coeffx(:,i);coeffy(:,i)]);
   end
   C(:,:,i) = psf2otf(kernel,params.N);
 end
 params.Csensval = C;
end

function y = CmE_sens(x,coeff,expleg_sens)
  y  = -feval(expleg_sens,x,coeff(2:end));
  y = cat(3,ones(size(x)),y);
end
