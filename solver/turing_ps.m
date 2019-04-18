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

rng(1);
u0 = u0 + sigma(1)*randn(N);
v0 = v0 + sigma(2)*randn(N);
yu0 = fftn(u0);
yv0 = fftn(v0);
y0 = [yu0(:); yv0(:)];

[k2,k] = formk(N,L);

JD = -k2(:)*D;
J = JD(:) + reshape(ones(n,1)*A2,[],1);

rungpu = true;
if rungpu
  y0 = gpuArray(y0);
  A1 = gpuArray(A1);
  b = gpuArray(b);
  lb = gpuArray(lb);
  ub = gpuArray(ub);
  J = gpuArray(J);
  k2 = gpuArray(k2);
end

saveresult = false;
if saveresult
  filepath = '/home/hbozhao/Dropbox (MIT)/2.168 Project/Data/turing';
  mat = matfile(filepath,'Writable',true);
  mat.A1 = [];
  mat.y = [];
end
Nsave = 1000;
saveind = 0;

Nt = 1100;
dt = 0.6;
outputstep = 200:100:Nt;
dt = 0.2;
Nt = 5000;
outputstep = [1,100:100:1000] ;%[300,1000,2000:1000:Nt];
nall = 1;
thresh = n*1e-5;
termination = @(t,y) event_gradient(t,y,k2(:),thresh);
if saveresult
  transform = @(y) [reshape(real(ifftn(reshape(y(1:n),N))),[],1);reshape(real(ifftn(reshape(y(n+(1:n)),N))),[],1)];
else
  transform = [];
end
naccept = 0;
ind = 0;
if saveresult
  nblock = Nsave;
else
  nblock = nall;
end
Anorm = norm(A1+diag(A2),inf);
A1 = reshape(A1,1,[]);
nparams = length(A1);
sigma = 0.05*eye(nparams);
if saveresult
  yall = zeros(nblock,N(1),N(2),length(outputstep),2,'gpuArray');
else
  yall = zeros(nblock,n*2,length(outputstep),'gpuArray');
end
A1all = zeros(nblock,nparams,'gpuArray');
tic;
while naccept < nall
  A1new = A1 + mvnrnd(zeros(1,nparams),sigma);
  A1new = reshape(A1new,2,2);
  Anew = A1new+diag(A2);
  if LSA(Anew,D) && (norm(Anew,inf)<5*Anorm)
    fnlin = @(t,y) turing_nlin_bd(t,y,A1new,b,lb,ub,N);
    [~,y,et] = odeimexez(fnlin,J,dt,Nt,y0,[],outputstep,termination,rungpu,transform);
    if ~et
      A1 = reshape(A1new,1,[]);
      naccept = naccept + 1;
      ind = ind + 1;
      if saveresult
        yall(ind,:,:,:,:) = permute(reshape(y,N(1),N(2),2,[]),[1,2,4,3]);
      else
        yall(ind,:,:) = y;
      end
      A1all(ind,:) = A1;
      runtime=toc;
      disp(['case ',num2str(naccept),' finished: Elapsed time (min): ',num2str(runtime/60),' ETA (min): ',num2str(runtime/60*(nall-naccept)/naccept)]);
      if saveresult && mod(naccept,Nsave)==0
        if naccept == Nsave
          mat.A1 = gather(A1all);
          mat.y = gather(yall);
        else
          mat.A1(saveind+(1:Nsave),:) = gather(A1all);
          mat.y(saveind+(1:Nsave),:,:,:,:) = gather(yall);
        end
        yall = zeros(nblock,N(1),N(2),length(outputstep),2,'gpuArray');
        A1all = zeros(nblock,nparams,'gpuArray');
        saveind = saveind+Nsave;
        ind = 0;
      end
    end
  end
end

if ~saveresult
  figure;
  if ndims(yall) == 3
    k2real(reshape(permute(yall(1:naccept,1:n,:),[1,3,2]),[],n).',N,[],'colorbar','on');
  else
    k2real(yall(1:naccept,1:n,:).',N);%,[],'GridSize',[9,9]);
  end
end
