%based on DDFT_nucleation11
%same time range as 13, a different initial guess 
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

tspan2 = linspace(0,2.5,100);
[t2,y2] = solver_DDFT(tspan2,y0,params);

ind = 1:100;
tdata = t2(ind);
ydata = y2(ind,:);
toc


kernelSize = 2;
Cspace = 'isotropic';
params.moreoptions = moreodeset('gmresTol',1e-5);


resultpath = [largedatapath,'DDFT_nucleation15.mat'];

options = optimoptions('fminunc','OutputFcn', @(x,optimvalues,state) save_opt_history(x,optimvalues,state,resultpath));
options = optimoptions(options,'HessianFcn','objective','Algorithm','trust-region');

x_guess = [0,-12,0.5,0,-3];





[x_opt,~,exitflag,params] = IP_DDFT(tdata,ydata,params,kernelSize,Cspace,options,x_guess,'Nmu',3,'tspan',300);



% ypred = IP_DDFT(tdata,ydata,params,kernelSize,Cspace,[],x_opt,'Nmu',3,'eval',true);
