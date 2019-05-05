function [tout,yout,et] = odeimex(fnlin,J,h,Nt,y0,pencil,outputstep,termination,gpu,transform)
%easy version odeimex: fixed time step, no error control
%h: step size
%outputstep: an array that indicates at which step the solution is outputed (must be sorted). By default, it only returns the last step
%termination: a function handle (t,y,yp) that returns true or a nonzero value when the solver should abort the iteration. The output et is true or the corresponding value when it's terminated early. By default, termination is empty
%when the solver terminates, the last column of yout is the y at the time step of termination, the unfilled yout is removed. Similarly for tout
%y is the value at the latest time point, yp is the estimated time derivative (y-yold)/dt
%the output yout is stored on gpu if gpu is true. For now, tout is always stored on cpu. By default, gpu is set to false.
%transform is a function handle that transforms the output y (by default empty)
%04/11/2019, now J can be a matrix or a column vector whose size is the same as y
%pencil(b,hinvGak) should return (Mt-hinvGak*J)\b
%right now J is constant
neq = length(y0);
y = y0;
t = 0:h:(h*Nt);
if nargin < 7 || isempty(outputstep)
  outputstep = Nt;
end
if nargin < 8
  termination = [];
end
if nargin < 9 || isempty(gpu)
  gpu = false;
end
if nargin < 10
  transform = [];
end
tout = t(outputstep);
if gpu
  yout = zeros(neq,length(outputstep),'gpuArray');
else
  yout = zeros(neq,length(outputstep));
end
outputid = 1;
if outputstep(outputid) == 1
  if isempty(transform)
    yout(:,outputid) = y;
  else
    yout(:,outputid) = transform(y);
  end
  outputid = outputid + 1;
end
et = false;

vectorJ = all(size(J)==size(y));

for step = 2:Nt
  yold = y;
  fex = fnlin(t(step-1),y);                     % the nonlinear part is fixed in the iteration
  if vectorJ
    y = (y+h*fex)./(1-h*J);
  else
    y = pencil(y+h*fex,h);
  end
  if outputstep(outputid) == step
    if isempty(transform)
      yout(:,outputid) = y;
    else
      yout(:,outputid) = transform(y);
    end
    outputid = min(outputid+1,length(outputstep));
  end
  if ~isempty(termination)
    yp = (y-yold)/h;
    et = termination(t(step),y,yp);
    if et
      yout(:,outputid) = y;
      yout(:,outputid+1:end) = [];
      tout(:,outputid) = h*(step-1);
      tout(:,outputid+1:end) = [];
      break;
    end
  end
end
