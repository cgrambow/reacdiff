function [x_opt,fval,exitflag] = IP_DDFT(tdata,ydata,params,kernelSize,Cspace,options,x_guess,mode)
%set mode = 'eval' to only do the forward solve and return
addpath('../../CHACR/GIP')

% params.N = [256,256];
% params.L = [5,5];
% params.dx = params.L ./ params.N;
if isequal(Cspace,'isotropic')
  [k2,~] = formk(params.N,params.L);
  k = sqrt(k2);
%   bound = [0, 2*pi*params.N(1)/params.L(1)/2]; %assuming at least 5 points across each pattern
  bound = [0,max(k(:))];
  expleg = custom_Legendre(1,bound);
  params.Cfunc.func = @(x) expleg.func(k,x);
  params.Cfunc.sens = @(x) expleg.sens(k,x);
  NC = kernelSize;
else
  NC = floor((prod(kernelSize)+1)/2);
  params.Csens = @(y,i) Csens_ASA(y,i,Cspace,kernelSize);
end

meta.C.index = 1:NC;
meta.C.exp = false;
if nargin < 7 || isempty(x_guess)
  x_guess = zeros(1,NC);
end

tspan = 100;

if nargin > 7 && isequal(mode,'eval')
  x_opt = IP(tdata,ydata,x_guess,meta,params, ...
  @(tdata,y0,FSA,meta,params) forwardSolver(tdata,y0,FSA,meta,params,tspan), ...
  [], [], [],...
  @(name,xparam,params) assign(name,xparam,params,Cspace,kernelSize),'eval',true);
else
  loss = @(y,ydata,~) MSE(y,ydata,prod(params.dx)*100);
  lossHess = @(dy,~,~,~) MSE([],[],prod(params.dx)*100,dy);
  if nargin > 5 && ~isempty(options)
    options = optimoptions(options,'SpecifyObjectiveGradient',true,'Display','iter-detailed');
  else
    options = optimoptions('fminunc','SpecifyObjectiveGradient',true,'Display','iter-detailed');
  end
  [x_opt,fval,exitflag] = fminunc(@(x) IP(tdata,ydata,x,meta,params, ...
  @(tdata,y0,FSA,meta,params) forwardSolver(tdata,y0,FSA,meta,params,tspan), ...
  @adjointSolver, ...
  loss,lossHess, ...
  @(name,xparam,params) assign(name,xparam,params,Cspace,kernelSize)), ...
  x_guess, options);
end

end

function params = assign(name,xparam,params,Cspace,kernelSize)
  switch name
  case 'mu'
    params.(name).params = xparam;
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
    if isequal(Cspace,'isotropic')
      %C(k) can be represented by a 1D function
      params.C = params.Cfunc.func(xparam);
      params.Csensval = params.Cfunc.sens(xparam);
    else
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

function [tout,y,dy,params] = forwardSolver(tdata,y0,FSA,meta,params,tspan)
  %tspan can either be sol or number of time points
  if any(params.C(:)>2)
    tout = NaN;
    y = NaN;
    dy = NaN;
  end
  if isequal(tspan,'sol')
    sol = true;
  else
    sol = false;
    tdata = linspace(tdata(1),tdata(end),tspan);
  end
  [tout,y,params,dy] = solver_DDFT(tdata,y0,params,meta,'forward',sol,FSA);
end

function grad = adjointSolver(tdata,sol,lossgrad,meta,discrete,~,params)
  grad = solver_DDFT(tdata,sol,params,meta,'adjoint',lossgrad,discrete);
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
