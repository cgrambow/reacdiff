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

rng(1);
reinitialize = true;
if ~reinitialize
  u0 = u0 + sigma(1)*randn(N);
  v0 = v0 + sigma(2)*randn(N);
  yu0 = fftn(u0);
  yv0 = fftn(v0);
  y0 = [yu0(:); yv0(:)];
end

[k2,k] = formk(N,L);

JD = -k2(:)*D;
J = JD(:) + reshape(ones(n,1)*A2,[],1);


saveresult = false;
if saveresult
  filepath = '/home/hbozhao/Dropbox (MIT)/2.168 Project/Data/turing_pca';
  mat = matfile(filepath,'Writable',true);
  mat.A1 = [];
  mat.y = [];
end
Nbatch = 250;
nall = 10000;
%Nbatch = 2;
%nall = 16;
nbatch = nall/(Nbatch*4); %must be integer

Nt = 1000;
dt = 1;
outputstep = Nt;
thresh = n*1e-5;
termination = @(t,y,~) event_gradient(t,y,k2(:),thresh);
if saveresult
  transform = @(y) [reshape(real(ifftn(reshape(y(1:n),N))),[],1);reshape(real(ifftn(reshape(y(n+(1:n)),N))),[],1)];
else
  transform = [];
end
Anorm = norm(A1+diag(A2),inf);
A1 = reshape(A1,1,[]);
nparams = length(A1);
sigma = 0.05*eye(nparams);

rungpu = false;

tic;
for batch = 1:nbatch
  ybatchall = zeros(4,Nbatch,N(1),N(2),length(outputstep),2);
  A1batchall = zeros(4,Nbatch,nparams);
  parfor process = 1:4
    ind = 0;
    ybatch = zeros(Nbatch,N(1),N(2),length(outputstep),2);
    A1batch = zeros(Nbatch,nparams);
    while ind < Nbatch
      A1new = A1lb + (A1ub-A1lb).*rand(2);
      Anew = A1new+diag(A2);
      if LSA(Anew,D) && (norm(Anew,inf)<5*Anorm)
        fnlin = @(t,y) turing_nlin_bd(t,y,A1new,b,lb,ub,N);
        if reinitialize
          y0 = [reshape(fftn(u0 + sigma(1)*randn(N)),[],1); reshape(fftn(v0 + sigma(2)*randn(N)),[],1)];
        end
        [~,y,et] = odeimexez(fnlin,J,dt,Nt,y0,[],outputstep,termination,rungpu,transform);
        if ~et
          A1 = reshape(A1new,1,[]);
          ind = ind + 1;
          ybatch(ind,:,:,:,:) = permute(reshape(y,N(1),N(2),2,[]),[1,2,4,3]);
          A1batch(ind,:) = A1;
        end
      end
    end
    A1batchall(process,:,:) = A1batch;
    ybatchall(process,:,:,:,:,:) = ybatch;
  end
  if saveresult
    if batch == 1
      mat.A1 = reshape(A1batchall,[],nparams);
      mat.y = reshape(ybatchall,[],N(1),N(2),length(outputstep),2);
    else
      range = (batch-1)*Nbatch*4+(1:Nbatch*4);
      mat.A1(range,:) = reshape(A1batchall,[],nparams);
      mat.y(range,:,:,:,:) = reshape(ybatchall,[],N(1),N(2),length(outputstep),2);
    end
    runtime=toc;
    disp(['batch ',num2str(batch),' of size ',num2str(Nbatch*4),' saved: Elapsed time (min): ',num2str(runtime/60),' ETA (min): ',num2str(runtime/60*(nbatch-batch)/batch)]);
  end
end

%plot 9 random realizations
figure;
for i=1:9
  subplot(3,3,i);
  imagesc(squeeze(mat.y(randi(nall),1:N(1),1:N(2),length(outputstep),1)));
  colormap gray;
  axis equal;
  ax = gca;
  ax.XTick=[]; ax.YTick=[];
end
saveas(gcf,[filepath,'_sample'],'png');
