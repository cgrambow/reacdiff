%parameter search
L = [5,5]*2;
N = [256,256];
n = prod(N);
n0 = 0.07;
sigma = 0.01;

rng(1);
n0 = n0 + sigma*randn(N);
y0 = fftn(n0);
y0 = y0(:);


[k2,k] = formk(N,L);
k2 = k2(:);
kind1 = k{1}; kind2 = k{2};
[kind1,kind2] = ndgrid(kind1(:),kind2(:));
theta = angle(kind1+i*kind2);

theta = theta(:);

k0 = 10;
alpha = 2;
C2 = exp(-(sqrt(k2)-k0).^2/(2*alpha^2));
J = -k2.*(-C2);

tspan = linspace(0,1.08,100);
options = odeset('AbsTol',1e-3);
[tout,yout] = odeimex(@(t,y) DDFT_nlin(t,y,k2,N),J,tspan,y0);

y0 = ifftn(reshape(yout(:,end),N));

xx = linspace(0,L(1),N(1));
yy = linspace(0,L(2),N(2));
[xx,yy] = meshgrid(xx,yy);

roi = 1./(1+exp(-(xx-0.64*L(1))/0.1)) .* 1./(1+exp((yy-0.4*L(2))/0.1));
rho = (sum(y0(:)) - sum(roi(:).*y0(:))) / sum(1-roi(:));
y0new = roi.*y0 + (1-roi)*rho;
%interpolate
y0new = imresize(y0new,2);
y0new = fftn(y0new);
y0new = y0new(:);
N = [256,256]*2;
figure; k2real(y0new,N);

save('DDFT_nucleus','y0new');

tspan2 = linspace(0,1,100);
options = odeset('AbsTol',1e-3);
[tout2,yout2] = odeimex(@(t,y) DDFT_nlin(t,y,k2,N),J,tspan2,y0new);
