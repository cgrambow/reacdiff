%parameter search
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
A1lb = [-0.5,-1.5;-0.5,-1.5];
A1ub = [1.5,0.5;1.5,0.5];
mat = matfile('/home/hbozhao/Dropbox (MIT)/2.168 Project/Data/turing_ss.mat');

rng(1);
u0 = u0 + sigma(1)*randn(N);
v0 = v0 + sigma(2)*randn(N);
yu0 = fftn(u0);
yv0 = fftn(v0);
y0 = [yu0(:); yv0(:)];

[k2,k] = formk(N,L);

JD = -k2(:)*D;
J = JD(:) + reshape(ones(n,1)*A2,[],1);

rungpu = false;
Nt = 1000;
dt = 1;
outputstep = floor(linspace(1,Nt,300));
transform = @(y) [reshape(real(ifftn(reshape(y(1:n),N))),[],1);reshape(real(ifftn(reshape(y(n+(1:n)),N))),[],1)];

for ind = 2%1:2
  switch ind
  case 1
    A1 = [0.08,-0.08;0.1,0];
  case 2
    A1 = reshape(mat.A1(1021,1:4),2,2);
  end
  fnlin = @(t,y) turing_nlin_bd(t,y,A1,b,lb,ub,N);
  [t,y,et] = odeimexez(fnlin,J,dt,Nt,y0,[],outputstep,[],rungpu,transform);
  y = reshape(y,N(1),N(2),2,[]);
  ymax = max(max(max(y,[],1),[],2),[],4);

  video_file = ['/home/hbozhao/Dropbox (MIT)/MIT Courses/2.168/2.168 Project/Final/movie_',num2str(ind)];
  myMovie = VideoWriter(video_file);
  set(myMovie,'FrameRate',30);
  open(myMovie);
  for i = 1:length(t)
    for j = 1:2
      subplot(1,2,j)
      imagesc(y(:,:,j,i));
      a = gca;
      a.XTick = []; a.YTick = [];
      axis tight; axis equal;
      a.CLim = [0,ymax(j)];
      colormap('gray')
      colorbar;
    end
    F = print('-RGBImage');
    writeVideo(myMovie,F);
    delete(findall(gcf,'Type','Axes'));
  end
  close(myMovie);
end
