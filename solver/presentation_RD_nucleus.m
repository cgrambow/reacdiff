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

imcenter = y1((N(1)/2-50):(N(1)/2+50),(N(2)/2-50):(N(2)/2+50));
imcenter = y1(1:101,1:101);
[ii,jj] = find(imcenter==max(imcenter(:)));
xx = 1:size(imcenter,1);
yy = 1:size(imcenter,2);
[xx,yy]=ndgrid(xx,yy);
%mask around the nucleus
mask = ((xx-ii).^2+(yy-jj).^2)<(pi/k0*N(1)/L(1))^2;
nucleus = imcenter .* mask;
ind1 = find(any(mask,2));
ind2 = find(any(mask,1));
nucleus = nucleus(ind1,ind2);
mask = mask(ind1,ind2);
shift = floor(size(mask)/2);
roi = zeros(N);
y0 = zeros(N);
xind = N(1)/2-shift(1)-1 + (1:size(mask,1));
yind = N(2)/2-shift(2)-1 + (1:size(mask,2)); %indices for placing nucleus in the center
roi(xind,yind) = mask;
y0(xind,yind) = nucleus;

%initial condition for the crystallization from a nuclues
y0 = y0(:);
roi = roi(:);
y02 = 0.08;
rho = (y02*n - sum(roi.*y0)) / sum(1-roi);
y0 = roi.*y0 + (1-roi)*rho;

tspan = linspace(0,0.4,601);
[t2,y2,params] = solver_DDFT(tspan,y0,params);
y2 = permute(reshape(y2,[length(t2),N]),[2,3,1]);

ysample = y2(:,:,1:60:end);
% imwrite((y2-min(y2(:)))/(max(y2(:))-min(y2(:))),[largedatapath,'presentation_RD_nucleus.png']);

mv = VideoWriter([largedatapath,'presentation_RD_nucleus']);
open(mv);
writeVideo(mv,(permute(y2,[1,2,4,3])-min(y2(:)))/(max(y2(:))-min(y2(:))));
close(mv);

mv = VideoWriter([largedatapath,'presentation_RD_nucleus']);
open(mv);
yk = abs(fft2(y2));
yk(1,1,:) = 0;
yk = fftshift(fftshift(yk,1),2);
Ncutoff = k0.*L/2/pi;
Ncutoff = floor(Ncutoff * 4);
yk = yk((N(1)/2-Ncutoff):(N(1)/2+Ncutoff),(N(2)/2-Ncutoff):(N(2)/2+Ncutoff),:);
writeVideo(mv,permute(min(yk./max(yk,[],[1,2])*10,1),[1,2,4,3]));
close(mv);
