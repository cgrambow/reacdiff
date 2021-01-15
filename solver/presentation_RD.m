%L = [5,5];
%N = [1000,1000];
L = [10,10];
N = [2000,2000];
%N = [256*2,256*2];
%L = [2.5,10];
%N = [500,2000];
%L = [5,5];
%N = [2000,2000];
n = prod(N);

params.N = N;
params.L = L;

[k2,k] = formk(N,L);
k0 = 50;
%R_D5
%k0 = 100;
alpha = 5;
params.C = exp(-(sqrt(k2)-k0).^2/(2*alpha^2))*0.95;


tspan = 0:0.1:5;
tspan = linspace(0,0.2,101);
params.options = odeset('OutputFcn',@(t,y,flag) simOutput(t,y,flag,true,[]));

y0 = randomfield(N,L,'RBF',2*pi/k0/10,1);
y0 = y0(:);
y0 = [];
[t1,y1,params] = solver_DDFT(tspan,y0,params);

y = permute(reshape(y1,[length(tspan),N]),[2,3,1]);


%save([largedatapath,'presentation_RD_4'],'y');
% im = circshift(circshift(y(:,:,18),300,2),250,1);
% im = (im-min(im(:)))/(max(im(:))-min(im(:)));
% im = ind2rgb(floor(im*255),cmocean('thermal'));
% imwrite(im,'C:\Users\zhbkl\Dropbox (MIT)\Research\Report 6\figure\presentation_RD.png');
