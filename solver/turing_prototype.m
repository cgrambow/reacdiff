%script for prototyping
L = [60,60]/2;
N = [256,256]/2;
n = prod(N);
D = [0.02,0.5];
b = [0.04,-0.15];
A1 = [0.08,-0.08;0.1,0];
A2 = -[0.03,0.08];
lb = [0,0];
ub = [0.2,0.5];
u0 = 5;
v0 = 5;
sigma = [0.01,0.01];

rng(4);
u0 = u0 + sigma(1)*randn(N);
v0 = v0 + sigma(2)*randn(N);
yu0 = fftn(u0);
yv0 = fftn(v0);
y0 = [yu0(:); yv0(:)];

[k2,k] = formk(N,L);

JD = -k2(:)*D;
J = JD(:) + reshape(ones(n,1)*A2,[],1);

Nt = 10000;
dt = 1;
outputstep = Nt;
thresh(1) = n*1e-5; %threshold for L2 norm of gradient
thresh(2) = 2;%about n*1e-3/8; %threshold for L2 norm of time derivative
termination = @(t,y,yp) event_gradient_ss(t,y,yp,k2(:),thresh);
fnlin = @(t,y) turing_nlin_bd(t,y,A1,b,lb,ub,N);
[t,y,et] = odeimexez(fnlin,J,dt,Nt,y0,[],outputstep,termination);

figure;
k2real(y(1:n,:),N);
