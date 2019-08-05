addpath('../../CHACR/GIP')
Nbasis = 5000;
t = 1;
k = linspace(0,100,10000)';
k2 = k.^2;
k0 = 10;
alpha = 5;
C = exp(-(k-k0).^2/(2*alpha^2))*0.95;
y0 = 0.06;
dmu = (1-y0+y0^2);
sigma = - k2 .* (dmu - C);

bound = [k(1),k(end)];
kscale = (k-mean(bound))/diff(bound);
psi = legendrepoly(kscale,Nbasis) .* sqrt(2*(1:Nbasis)-1);
% psi = laguerrepoly(k,Nbasis) .* exp(-k/2);
deta = k .* k2 .* exp(sigma*t) .* psi;
scale = (k .* k2 .* exp(sigma*t)).^2;
hessian = deta' * deta;
[V,D] = eig(hessian);

xx = linspace(0,20,1000)';
figure;
for i = 1:3
  subplot(1,3,i);
  % plot(xx,laguerrepoly(xx,[],V(:,end-i+1)).*exp(-xx/2));
  % plot(k,psi*V(:,end-i+1));
  plot(xx,legendrepoly((xx-mean(bound))/diff(bound),[],V(:,end-i+1) .* sqrt(2*(1:Nbasis)-1)'));
end

figure;
plot(k,psi*V(:,end));