function [tout,yout] = odeimex(fnlin,J,Nt,y0,options,pencil)
%a simplified ode solver adapted from ode15s
%the backward diffusion / NDF is applied for the linear part of the equation, in particular, the propotionality constant is J, note that J is the array the same as y0 (equivalent to diagonal) for now, in the near future, generalize J to a matrix.
%The nonlinear part of the equation is explicit, that is, at each time step, this term is dependent only on the previous time step.
%only ODE (not DAE) is supported right now.
%dy/dt = J*y + fnlin
%Nt is the number of time steps
%the format of yout is different from typical ode, each column corresponds to a time point given in tout, which is a column vector.
%03/25/2019, now J can be a matrix or a column vector whose size is the same as y
%pencil(b,hinvGak) should return (Mt-hinvGak*J)\b
%right now J is constant
if nargin<5
  options = [];
end
maxk = odeget(options,'MaxOrder',5,'fast');
bdf = strcmp(odeget(options,'BDF','off','fast'),'on');
htry = abs(odeget(options,'InitialStep',[],'fast'));
hmax = abs(odeget(options,'MaxStep',inf,'fast'));
rtol = odeget(options,'RelTol',1e-3,'fast');
atol = odeget(options,'AbsTol',1e-6,'fast');
normcontrol = strcmp(odeget(options,'NormControl','off','fast'),'on');

neq = length(y0);

threshold = atol / rtol;
if normcontrol
  normy = norm(y0);
end

tdir = 1;
t = 0;
y = y0;
tout = zeros(Nt,1);
yout = zeros(neq,Nt);
tout(1) = t;
yout(:,1) = y;
vectorJ = all(size(J)==size(y));
if vectorJ
  yp = fnlin(t,y) + J.*y;
else
  yp = fnlin(t,y) + J*y;
end

% Initialize method parameters.
G = [1; 3/2; 11/6; 25/12; 137/60];
if bdf
  alpha = [0; 0; 0; 0; 0];
else
  alpha = [-37/200; -1/9; -0.0823; -0.0415; 0];
end
invGa = 1 ./ (G .* (1 - alpha));
erconst = alpha .* G + (1 ./ (2:6)');
difU = [ -1, -2, -3, -4,  -5;           % difU is its own inverse!
          0,  1,  3,  6,  10;
          0,  0, -1, -4, -10;
          0,  0,  0,  1,   5;
          0,  0,  0,  0,  -1 ];
maxK = 1:maxk;
[kJ,kI] = meshgrid(maxK,maxK);
difU = difU(maxK,maxK);

% Determine initial step size

% hmin is a small number such that t + hmin is clearly different from t in
% the working precision, but with this definition, it is 0 if t = 0.
hmin = 16*eps*abs(t);

if isempty(htry)
  % Compute an initial step size h using yp = y'(t).
  if normcontrol
    wt = max(normy,threshold);
    rh = 1.25 * (norm(yp) / wt) / sqrt(rtol);  % 1.25 = 1 / 0.8
  else
    wt = max(abs(y),threshold);
    rh = 1.25 * norm(yp ./ wt,inf) / sqrt(rtol);
  end
  absh = hmax;
  if absh * rh > 1
    absh = 1 / rh;
  end
  absh = max(absh, hmin);
else
  absh = min(hmax, max(hmin, htry));
end
h = tdir * absh;

% Initialize.
k = 1;                                  % start at order 1 with BDF1
K = 1;                                  % K = 1:k
klast = k;
abshlast = absh;

dif = zeros(neq,maxk+2);
dif(:,1) = h * yp;

hinvGak = h * invGa(k);
nconhk = 0;                             % steps taken with current h and k

% THE MAIN LOOP

at_hmin = false;
for step = 2:Nt

  hmin = 16*eps(t);
  absh = min(hmax, max(hmin, absh));
  if absh == hmin
    if at_hmin
      absh = abshlast;  % required by stepsize recovery
    end
    at_hmin = true;
  else
    at_hmin = false;
  end
  h = tdir * absh;

  if (absh ~= abshlast) || (k ~= klast)
    difRU = cumprod((kI - 1 - kJ*(absh/abshlast)) ./ kI) * difU;
    dif(:,K) = dif(:,K) * difRU(K,K);

    hinvGak = h * invGa(k);
    nconhk = 0;

  end

  % LOOP FOR ADVANCING ONE STEP.
  nofailed = true;                      % no failed attempts
  fex = fnlin(t,y);                     % the nonlinear part is fixed in the iteration
  while true                            % Evaluate the formula.
    % Compute the constant terms in the equation for ynew.
    psi = dif(:,K) * (G(K) * invGa(k));
    % Predict a solution at t+h.
    tnew = t + h;
    pred = y + sum(dif(:,K),2);
    ynew = pred;

    if normcontrol
      normynew = norm(ynew);
      invwt = 1 / max(max(normy,normynew),threshold);
    else
      invwt = 1 ./ max(max(abs(y),abs(ynew)),threshold);
    end

    if vectorJ
      ynew2 = (ynew-psi+hinvGak*fex)./(1-hinvGak*J);
    else
      ynew2 = pencil(ynew-psi+hinvGak*fex,hinvGak);
    end

    difkp1 = ynew2-ynew;
    ynew = ynew2;

    % difkp1 is now the backward difference of ynew of order k+1.
    if normcontrol
      err = (norm(difkp1) * invwt) * erconst(k);
    else
      err = norm(difkp1 .* invwt,inf) * erconst(k);
    end

    if err > rtol                       % Failed step
      if absh <= hmin
        warning(message('MATLAB:ode15s:IntegrationTolNotMet', sprintf( '%e', t ), sprintf( '%e', hmin )));
        tout(step:end) = [];
        yout(step:end,:) = [];
        return;
      end

      abshlast = absh;
      if nofailed
        nofailed = false;
        hopt = absh * max(0.1, 0.833*(rtol/err)^(1/(k+1))); % 1/1.2
        if k > 1
          if normcontrol
            errkm1 = (norm(dif(:,k) + difkp1) * invwt) * erconst(k-1);
          else
            errkm1 = norm((dif(:,k) + difkp1) .* invwt,inf) * erconst(k-1);
          end
          hkm1 = absh * max(0.1, 0.769*(rtol/errkm1)^(1/k)); % 1/1.3
          if hkm1 > hopt
            hopt = min(absh,hkm1);      % don't allow step size increase
            k = k - 1;
            K = 1:k;
          end
        end
        absh = max(hmin, hopt);
      else
        absh = max(hmin, 0.5 * absh);
      end
      h = tdir * absh;

      difRU = cumprod((kI - 1 - kJ*(absh/abshlast)) ./ kI) * difU;
      dif(:,K) = dif(:,K) * difRU(K,K);

      hinvGak = h * invGa(k);
      nconhk = 0;

    else                                % Successful step
      break;

    end
  end % while true

  dif(:,k+2) = difkp1 - dif(:,k+1);
  dif(:,k+1) = difkp1;
  for j = k:-1:1
    dif(:,j) = dif(:,j) + dif(:,j+1);
  end

  klast = k;
  abshlast = absh;
  nconhk = min(nconhk+1,maxk+2);
  if nconhk >= k + 2
    temp = 1.2*(err/rtol)^(1/(k+1));
    if temp > 0.1
      hopt = absh / temp;
    else
      hopt = 10*absh;
    end
    kopt = k;
    if k > 1
      if normcontrol
        errkm1 = (norm(dif(:,k)) * invwt) * erconst(k-1);
      else
        errkm1 = norm(dif(:,k) .* invwt,inf) * erconst(k-1);
      end
      temp = 1.3*(errkm1/rtol)^(1/k);
      if temp > 0.1
        hkm1 = absh / temp;
      else
        hkm1 = 10*absh;
      end
      if hkm1 > hopt
        hopt = hkm1;
        kopt = k - 1;
      end
    end
    if k < maxk
      if normcontrol
        errkp1 = (norm(dif(:,k+2)) * invwt) * erconst(k+1);
      else
        errkp1 = norm(dif(:,k+2) .* invwt,inf) * erconst(k+1);
      end
      temp = 1.4*(errkp1/rtol)^(1/(k+2));
      if temp > 0.1
        hkp1 = absh / temp;
      else
        hkp1 = 10*absh;
      end
      if hkp1 > hopt
        hopt = hkp1;
        kopt = k + 1;
      end
    end
    if hopt > absh
      absh = hopt;
      if k ~= kopt
        k = kopt;
        K = 1:k;
      end
    end
  end
  % Advance the integration one step.
  t = tnew;
  y = ynew;
  if normcontrol
    normy = normynew;
  end
  tout(step) = t;
  yout(:,step) = y;
end
