%based on DDFT_nucleation37, sensitivity analysis, use Laguerre
addpath('../../CHACR/GIP')
runoptim = false;

resultpath = [largedatapath,'DDFT_nucleation38.mat'];

if runoptim
  tic;
  L = [5,5];
  N = [256,256];
  n = prod(N);

  params.N = N;
  params.L = L;
  [k2,k] = formk(N,L);
  k0 = 10;
  alpha = 5;
  params.C = exp(-(sqrt(k2)-k0).^2/(2*alpha^2))*0.95;

  [t1,y1,params] = solver_DDFT([],[],params);

  xx = linspace(0,L(1),N(1));
  yy = linspace(0,L(2),N(2));
  [xx,yy] = ndgrid(xx,yy);
  center = L/2;
  radius = 0.06*L(1);
  thickness = 0.01*L(1);
  roi = roi_circle(xx,yy,center,radius,thickness);

  %nucleus
  y0 = y1(end,:)';
  roi = roi(:);
  y02 = 0.045;
  rho = (y02*n - sum(roi.*y0)) / sum(1-roi);
  y0 = roi.*y0 + (1-roi)*rho;

  tspan2 = linspace(0,2.5,100);
  [t2,y2] = solver_DDFT(tspan2,y0,params);

  ind = 1:20:100;
  tdata = t2(ind);
  ydata = y2(ind,:);
  toc


  kernelSize = 300;
  Cspace = 'isotropic_laguerre_scale';
  params.moreoptions = moreodeset('gmresTol',1e-5);

  [hessian,hessian_t,dy] = IP_DDFT(tdata,ydata,params,kernelSize,Cspace,[],[],'discrete',true,'cutoff',k0,'assign_suppress',{'C'},'mode','sens');
  save(resultpath,'hessian','hessian_t','dy');
else
  addpath('../../External/boundedline-pkg')
  if ~exist('hessian','var')
    load(resultpath);
  end
  numBasis = 150;
  epsilon = 0.004*prod(params.L);
  x = linspace(0,5,1000)';
  psi = laguerrepoly(x,numBasis) .* exp(-x/2);
  C = psi * ((hessian(1:numBasis,1:numBasis)+epsilon*eye(numBasis))\psi');
  dev = diag(C);
  y = exp(-(x-1).^2/(2*(alpha/k0)^2))*0.95;
  figure;
  boundedline(x,y,dev/50);
  ylim([-0.5,1.5])
end
