%based on DDFT_nucleation11
%use a different time range
addpath('../../CHACR/GIP')
runoptim = false;

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

ind = 1:100;
tdata = t2(ind);
ydata = y2(ind,:);
toc

kernelSize = 2;
Cspace = 'isotropic';
params.moreoptions = moreodeset('gmresTol',1e-5);

resultpath = [largedatapath,'DDFT_nucleation13.mat'];

options = optimoptions('fminunc','OutputFcn', @(x,optimvalues,state) save_opt_history(x,optimvalues,state,resultpath));
options = optimoptions(options,'HessianFcn','objective','Algorithm','trust-region');

if runoptim
  x_guess = [0,-10,0.5,0,-3];
  [x_opt,~,exitflag,params] = IP_DDFT(tdata,ydata,params,kernelSize,Cspace,options,x_guess,'Nmu',3,'tspan',300);
else
  modelfunc.C = @(k) exp(-(k-k0).^2/(2*alpha^2))*0.95;
  modelfunc.mu = @(x) x - x.^2/2 + x.^3/3;
  arg.C = linspace(0,2,500);
  arg.mu = linspace(min(ydata(:)),max(ydata(:)),100);
  frameindex = 1:20:100;
  history_production(resultpath,[1,21,45,125],modelfunc,arg,tdata-tdata(1),ydata,params,kernelSize,Cspace,'FrameIndex',frameindex,'IP_DDFT_arg',{'Nmu',3},'yyaxisLim',[-1,1;-1,1],'k0',k0,'xlim',[min(ydata(:)),2]);
  f = gcf;
  f.Position = [680 337 522 641];
end
