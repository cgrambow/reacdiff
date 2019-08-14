%the script for the main figure for DDFT
addpath('../../CHACR/GIP')

L = [5,5];
N = [256,256];
n = prod(N);

params.N = N;
params.L = L;

[k2,k] = formk(N,L);
k0 = 10;
alpha = 5;
params.C = exp(-(sqrt(k2)-k0).^2/(2*alpha^2))*0.95;

if ~exist('t2','var')
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
end

rowtotal = 4;
columntotal = 5;
stparg = {0.05,[0.08,0.06],[0.07,0.03]};
clim = [min(min(ydata(:))),max(max(ydata(:)))];
%first row: data
h = visualize([],[],[],ydata,'c',false,'ImageSize',params.N,'caxis',clim,'GridSize',[1,NaN],'OuterGridSize',[rowtotal,1],'OuterSubplot',[1,1],'subtightplot',stparg);
for j = 1:length(h)
  title(h(j),['t = ',num2str(tdata(j),2)]);
end
axes(h(1));
text(0.01,1.55,'(a) Data','Units','normalized','HorizontalAlignment','left');
%second row: history of DDFT_nucleation23
kernelSize = 2;
Cspace = 'isotropic';
params.moreoptions = moreodeset('gmresTol',1e-5);
IP_DDFT_arg = {'Nmu',3};
figprop = {'IP_DDFT_arg',IP_DDFT_arg,'yyaxisLim',[-0.5,1.5;-1,1.2],'k0',k0,'xlim',[min(ydata(:)),2]};
resultpath = [largedatapath,'DDFT_nucleation23'];
varload = load(resultpath);
history = varload.history;
modelfunc.C = @(k) exp(-(k-k0).^2/(2*alpha^2))*0.95;
modelfunc.mu = @(x) x - x.^2/2 + x.^3/3;
arg.C = linspace(0,2,500);
arg.mu = linspace(min(ydata(:)),max(ydata(:)),100);
ind = [1,19,25,31,1797];
for i = 1:length(ind)
  subtightplot(rowtotal,columntotal,columntotal+i,stparg{:});
  [~,~,~,pp] = IP_DDFT(tdata,ydata,params,kernelSize,Cspace,[],history(ind(i),:),'mode','pp',IP_DDFT_arg{:});
  history_func(ind(i),modelfunc,arg,Cspace,pp,[],figprop{:});
  if i~=1
    xlabel([]);
    yyaxis left
    yticklabels([]);
    yyaxis right
    yticklabels([]);
  else
    text(0.01,1.55,'(b) Quartic approximation','Units','normalized','HorizontalAlignment','left');
    legend({'$\hat{C_2}(k)$ (truth)','$\hat{C_2}(k)$','$\mu_h(\eta)$ (truth)','$\mu_h(\eta)$'},...
    'Orientation','horizontal','Position',[0.10 0 0.8169 0.0374],'Interpreter','latex');
    legend('boxoff');
  end
end
%third row: final result of DDFT_nucleation23
if ~exist('yhistory','var')
  [yhistory,~,~,pp] = IP_DDFT(tdata,ydata,params,kernelSize,Cspace,[],history(end,:),'mode','eval',IP_DDFT_arg{:});
end
h = visualize([],[],[],yhistory,'c',false,'ImageSize',params.N,'caxis',clim,'GridSize',[1,NaN],'OuterGridSize',[rowtotal,1],'OuterSubplot',[3,1],'subtightplot',stparg);
text(h(1),0.01,1.2,'(c) Quartic approximation result','Units','normalized','HorizontalAlignment','left');

%fourth row: history of DDFT_nucleation30
kernelSize = 10;
Cspace = 'isotropic_hermite_scale';
Crange = [0,4];
IP_DDFT_arg = {'Nmu',0,'discrete',true,'cutoff',k0};
figprop = {'IP_DDFT_arg',IP_DDFT_arg,'yyaxisLim',[-0.5,1.2],'k0',k0,'xlim',Crange};
resultpath = [largedatapath,'DDFT_nucleation30'];
varload = load(resultpath);
history = varload.history;
arg.C = linspace(Crange(1),Crange(2),500);
ind = [1,11,16,21,36];
for i = 1:length(ind)
  subtightplot(rowtotal,columntotal,columntotal*3+i,stparg{:});
  if i==length(ind)
    DDFT_nucleation36;
    delete(hl);
    hold on;
  end
  [~,~,~,pp] = IP_DDFT(tdata,ydata,params,kernelSize,Cspace,[],history(ind(i),:),'mode','pp',IP_DDFT_arg{:});
  history_func(ind(i),modelfunc,arg,Cspace,pp,[],figprop{:});
  if i~=1
    xlabel([]);
    yticklabels([]);
  else
    text(0.01,1.55,'(d) Hermite function basis function','Units','normalized','HorizontalAlignment','left');
  end
end
f = gcf;
set(findall(f,'-property','FontName'),'FontName','Arial');
set(findall(f,'-property','FontWeight'),'FontWeight','normal');
set(findall(f,'-property','FontSize'),'FontSize',12);
f.Position = [680 298 560 680];
% print(f,'C:\Users\zhbkl\Dropbox (MIT)\Research\Report 6\figure\DDFT_mainFig','-dtiff','-r400');
