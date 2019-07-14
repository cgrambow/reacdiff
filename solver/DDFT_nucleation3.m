addpath('../../CHACR/GIP')
runoptim = true;

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

tspan2 = linspace(0,1.5,100);
[t2,y2] = solver_DDFT(tspan2,y0,params);

ind = 1:50;
tdata1 = t2(ind);
ydata1 = y2(ind,:);
ind = 50:100;
tdata2 = t2(ind);
ydata2 = y2(ind,:);
toc

kernelSize = [41,41];
Cspace = 'k';

resultpath = [largedatapath,'DDFT_nucleation3.mat'];
if runoptim
  save_history = true;
  options = optimoptions('fminunc','MaxIterations',5);
  if save_history
    options = optimoptions(options,'OutputFcn', @(x,optimvalues,state) save_opt_history(x,optimvalues,state,resultpath,[],true));
  end
  NC = floor((prod(kernelSize)+1)/2);
  x_opt = zeros(1,NC);
  while true
    x_opt = IP_DDFT(tdata1,ydata1,params,kernelSize,Cspace,options,x_opt,'tspan',200);
    x_opt = IP_DDFT(tdata2,ydata2,params,kernelSize,Cspace,options,x_opt,'tspan',200);
  end
else
  meta.C.index = floor((prod(kernelSize)+1)/2);
  meta.C.exp = false;
  frameindex = [1,6,12,23,41];
  history_production(resultpath,[1,2,11,71,171],[],[],meta,tdata-tdata(1),ydata,params,kernelSize,Cspace,'FrameIndex',frameindex,'CtruthSubplot',[2,length(frameindex)]);
  f = gcf;
  f.Position = [680 337 522 641];
end
