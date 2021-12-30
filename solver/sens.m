addpath('../../CHACR/GIP')
L = [5,5];
N = [256,256];
n = prod(N);
Nbasis = 1000;
t = 1;
[k2,k] = formk(N,L);
absk = sqrt(k2);
k0 = 10;
alpha = 5;
C = exp(-(absk-k0).^2/(2*alpha^2))*0.95;
y0 = 0.06;
dmu = (1-y0+y0^2);
sigma = - k2 .* (dmu - C);
psi = laguerrepoly(absk,Nbasis) .* exp(-absk/2);
deta = k2 .* exp(sigma*t) .* psi;
deta = reshape(deta,n,Nbasis);
hessian = deta' * deta;
[V,D] = eig(hessian);

xx = linspace(0,20,1000)';
figure;
for i = 1:3
  subplot(1,3,i);
  plot(xx,laguerrepoly(xx,[],V(:,end-i+1)).*exp(-xx/2));
end
