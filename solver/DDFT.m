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

% %use image?
% im = imread('cameraman.tif');
% im = imresize(im,N);
% C2 = fft(im);
% C2 = C2(:);
% C2(1) = 0;
% C2 = C2/max(abs(C2))/1.5;

[k2,k] = formk(N,L);
k2 = k2(:);
kind1 = k{1}; kind2 = k{2};
[kind1,kind2] = ndgrid(kind1(:),kind2(:));
theta = angle(kind1+i*kind2);

theta = theta(:);

k0 = 10;
alpha = 2;
C2 = exp(-(sqrt(k2)-k0).^2/(2*alpha^2));
C22 = C2;
% C22 = C2 .* (1+cos(theta))/2;
% C22 = C2 .* (exp(-(theta-pi/2).^2/(2*(50/180*pi)^2))+exp(-(theta+pi/2).^2/(2*(50/180*pi)^2)))/2;
J = -k2.*(-C22);

tspan = linspace(0,0.14,100);
tspan = [0,1.1];
% tspan = [0,0.14];
options = odeset('AbsTol',1e-3);
[tout,yout] = odeimex(@(t,y) DDFT_nlin(t,y,k2,N),J,tspan,y0);

figure; k2real(yout(:,end),N);
