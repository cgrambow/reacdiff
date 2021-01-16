%based on presentation_RD and DDFT_nucleation, start from a nucleus
L = [5,5];
N = [1024,1024];
% L = [10,10];
% N = [2000,2000];
n = prod(N);

params.N = N;
params.L = L;

[k2,k] = formk(N,L);
k0 = 50;
alpha = 5;
params.C = exp(-(sqrt(k2)-k0).^2/(2*alpha^2))*0.95;

params.options = odeset('OutputFcn',@(t,y,flag) simOutput(t,y,flag,true,[]));

tspan = linspace(0,0.2,3);

y0 = [];
[t1,y1,params] = solver_DDFT(tspan,y0,params);

y1 = y1(end,:);
y1 = reshape(y1,N);
%%
imcenter = y1((N(1)/2-50):(N(1)/2+50),(N(2)/2-50):(N(2)/2+50));
[ii,jj] = find(imcenter==max(imcenter(:)));
xx = 1:size(imcenter,1);
yy = 1:size(imcenter,2);
[xx,yy]=ndgrid(xx,yy);
%mask around the nucleus
mask = ((xx-ii).^2+(yy-jj).^2)<(pi/k0*N(1)/L(1))^2;
roi = zeros(N);
roi((N(1)/2-50):(N(1)/2+50),(N(2)/2-50):(N(2)/2+50)) = mask;

%initial condition for the crystallization from a nuclues
y0 = y1(:);
roi = roi(:);
y02 = 0.08;
rho = (y02*n - sum(roi.*y0)) / sum(1-roi);
y0 = roi.*y0 + (1-roi)*rho;

tspan = linspace(0,0.4,601);
[t2,y2,params] = solver_DDFT(tspan,y0,params);
y2 = permute(reshape(y2,[length(t2),N]),[2,3,1]);

ysample = y2(:,:,1:60:end);
% imwrite((y2-min(y2(:)))/(max(y2(:))-min(y2(:))),[largedatapath,'presentation_RD_nucleus.png']);
