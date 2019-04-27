L = [80,80];
N = 2^10*[1,1];
n = prod(N);
kappa = 1;
omega = 3;
sigma = 0.01;


rng(1);
y0 = 0.5 + sigma*randn(N);
y0 = fftn(y0);
y0 = y0(:);
[k2,k] = formk(N,L);
k2 = k2(:);

J = - kappa*k2.^2;

Nt = 15000;
dt = 0.01;
fps = 30;
outputstep = floor(linspace(1,Nt,fps*10));
fnlin = @(t,y) spdecomp_nlin(t,y,N,k2,omega);
transform = @(y) reshape(real(ifftn(reshape(y,N))),[],1);
[tout,yout] = odeimexez(fnlin,J,dt,Nt,y0,[],outputstep,[],false,transform);

myMovie = VideoWriter([largedatapath,'spdecomp_movie']);
set(myMovie,'FrameRate',fps);
open(myMovie);
writeVideo(myMovie,reshape(yout,N(1),N(2),1,[]));
close(myMovie);
