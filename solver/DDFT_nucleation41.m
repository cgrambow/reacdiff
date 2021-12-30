%based on DDFT_nucleation32
%change x_guess(1) to 0.1
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

ind = 1:20:100;
tdata = t2(ind);
ydata = y2(ind,:);
toc


kernelSize = 10;
Cspace = 'isotropic_hermite_scale';
params.moreoptions = moreodeset('gmresTol',1e-5);


resultpath = [largedatapath,'DDFT_nucleation41_ally.mat'];

options = optimoptions('fminunc','OutputFcn', @(x,optimvalues,state) save_opt_history(x,optimvalues,state,resultpath));
options = optimoptions(options,'HessianFcn','objective','Algorithm','trust-region','MaxFunctionEvaluations',10000,'MaxIterations',10000);

x_guess = [zeros(1,kernelSize),0.5,0,-3];
x_guess(1) = 0.1;

if runoptim
  [x_opt,~,exitflag,params] = IP_DDFT(tdata,ydata,params,kernelSize,Cspace,options,x_guess,'Nmu',3,'discrete',true,'cutoff',k0);
else
  modelfunc.C = @(k) exp(-(k-k0).^2/(2*alpha^2))*0.95;
  modelfunc.mu = @mu;
  Crange = [0,4];
  arg.C = linspace(Crange(1),Crange(2),500);
  arg.mu = linspace(min(ydata(:)),max(ydata(:)),100);
  % history_production(resultpath,[],modelfunc,arg,tdata-tdata(1),ydata,params,kernelSize,Cspace,...
  %     'IP_DDFT_arg',{'Nmu',3,'discrete',true,'cutoff',k0},'yyaxisLim',[-0.5,1.5;0,2],...
  %     'k0',k0,'xlim',[min(ydata(:)),Crange(2)],'showModelSolution',true,'offset',true,'muderiv',true,...
  %     'legend',{{'$\hat{C_2}(k)$ (truth)','$\hat{C_2}(k)$','$\mu_h(\eta)$ (truth)','$\mu_h(\eta)$'},...
  %     'Orientation','horizontal','Position',[0.2051 0.001 0.7874 0.0566],'Interpreter','latex'},...
  %     'FontSize',13,'stparg',{0.05,[0.1,0.08],[0.08,0.04]});
  % f = gcf;
  % f.Position = [680 368 588 610];

  figure('Color',[0,0,0],'Position',[593 189 1036 610],'InvertHardCopy','off');
  mv = VideoWriter('DDFT_nucleation41','MPEG-4');
  mv.FrameRate = 1;
  history_movie(resultpath,mv,[],modelfunc,arg,tdata-tdata(1),ydata,params,kernelSize,Cspace,...
      'yyaxisLim',[0.6,1.7;-0.2,1.2],'funcxlabel',false,'FontSize',14,...
      'visualizeArgs',{'edgeColor',[1,1,1],'edgeWidth',1,'colormap',colormap(gray)},...
      'k0',k0,'IP_DDFT_arg',{'Nmu',3,'discrete',true,'cutoff',k0},...
      'use_saved',true,'save',false,'muderiv',true,'offset',true,...
      'stparg',{[0.03,0.03],[0.05,0.15],0.08},'labelpos',[-0.3,1.6]);
end

function [y,dy] = mu(x)
  y = x - x.^2/2 + x.^3/3;
  dy = 1 - x + x.^2;
end
