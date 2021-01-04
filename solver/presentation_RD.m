L = [5,5];
N = [1000,1000];
L = [10,10];
N = [2000,2000];
n = prod(N);

params.N = N;
params.L = L;

[k2,k] = formk(N,L);
k0 = 50;
alpha = 5;
params.C = exp(-(sqrt(k2)-k0).^2/(2*alpha^2))*0.95;


tspan = 0:0.1:5;
[t1,y1,params] = solver_DDFT(tspan,[],params);


save([largedatapath,'presentation_RD'],'y');
% im = circshift(circshift(y(:,:,18),300,2),250,1);
% im = (im-min(im(:)))/(max(im(:))-min(im(:)));
% im = ind2rgb(floor(im*255),cmocean('thermal'));
% imwrite(im,'C:\Users\zhbkl\Dropbox (MIT)\Research\Report 6\figure\presentation_RD.png');