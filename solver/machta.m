n = 7;
theta = ones(1,n)/n;

ntime = n;
rholen = n^2;
theta = padarray(theta,[0,(rholen-n)/2],0,'both');
n = rholen;
center = ceil(rholen/2);
rho = zeros(ntime,rholen);
drho = zeros(ntime,rholen,n);
hessian_t = zeros(n,n,ntime);
rho(1,center) = 1;
for i = 2:ntime
  rho(i,:) = conv(rho(i-1,:),theta,'same');
  for j = 1:n
    drho(i,:,j) = conv(drho(i-1,:,j),theta,'same') + circshift(rho(i-1,:),j);
  end
  drho_t = squeeze(drho(i,:,:));
  hessian_t(:,:,i) = drho_t' * drho_t;
end
[V,D] = eig(hessian_t(:,:,end));
