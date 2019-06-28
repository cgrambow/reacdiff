%parameter search
L = [5,5];
N = [256,256]/2;
n = prod(N);
n0 = 0.07;
sigma = 0.01;

rng(1);
n0 = n0 + sigma*randn(N);
y0 = fftn(n0);
y0 = y0(:);

[k2,k] = formk(N,L);
k2 = k2(:);
k0 = 10;
alpha = 2;
C2 = exp(-(sqrt(k2)-k0).^2/(2*alpha^2));
J = -k2.*(-C2);

Nt = 10000;
options = odeset('AbsTol',1e-3);
[tout,yout] = odeimex(@(t,y) DDFT_nlin(t,y,k2,N),J,Nt,y0);

k2real(yout(:,end),N);
