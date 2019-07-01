function varargout = odeimex(fnlin,J,tspan,y0,options,pencil,moreoptions,JexMult,exsensFcn,dJdp)
%a simplified ode solver adapted from ode15s
%the backward diffusion / NDF is applied for the linear part of the equation, in particular, the propotionality constant is J, note that J can be an array the same as y0 (equivalent to diagonal), in this case, the mass matrix must be diagonal as well. Alternatively, J can be a matrix, in which case pencil is needed.
%The nonlinear part of the equation is explicit, that is, at each time step, this term is dependent only on the previous time step.
%only ODE (not DAE) is supported right now.
%dy/dt = J*y + fnlin
%Nt is the number of time steps
%the format of yout is different from typical ode, each column corresponds to a time point given in tout, which is a column vector.
%pencil(b,hinvGak) should return (Mt-hinvGak*J)\b
%right now J is constant
%the mass matrix Mt is also constant
%JexMult is the multiplying function for the jacobian of the explicit term (argument: ys, t, y)
%exsensFcn is the sensitivity of the explicit term
%dJdp is the sensitivity of the multiplying matrix of the implicit term


if nargin<5
  options = [];
end
if nargin<7
  moreoptions = [];
end
%output
FcnHandlesUsed  = true;
haveInterpFcn = false;
output_sol = (FcnHandlesUsed && (nargout==1) && ~haveInterpFcn);      % sol = odeXX(...)
output_ty  = (~output_sol && (nargout > 0) && ~haveInterpFcn);  % [t,y,...] = odeXX(...)
sol = []; kvec = []; dif3d = []; difs4d = [];
if output_sol
  sol.solver = solver_name;
  sol.extdata.odefun = ode;
  sol.extdata.options = options;
  sol.extdata.varargin = varargin;
end

maxk = odeget(options,'MaxOrder',5,'fast');
bdf = strcmp(odeget(options,'BDF','off','fast'),'on');
htry = abs(odeget(options,'InitialStep',[],'fast'));
hmax = abs(odeget(options,'MaxStep',inf,'fast'));
rtol = odeget(options,'RelTol',1e-3,'fast');
atol = odeget(options,'AbsTol',1e-6,'fast');
Mt = odeget(options,'Mass',[],'fast');
normcontrol = strcmp(odeget(options,'NormControl','off','fast'),'on');
%copied from odearguments
htspan = abs(tspan(2) - tspan(1));
tspan = tspan(:);
ntspan = length(tspan);
t0 = tspan(1);
next = 2;       % next entry in tspan
tfinal = tspan(end);
tdir = sign(tfinal - t0);
refine = max(1,odeget(options,'Refine',1,'fast'));
if ntspan > 2
  outputAt = 'RequestedPoints';         % output only at tspan points
elseif refine <= 1
  outputAt = 'SolverSteps';             % computed points, no refinement
else
  outputAt = 'RefinedSteps';            % computed points, with refinement
  S = (1:refine-1) / refine;
end
idxNonNegative = [];

%Forward sensitivity analysis
FSA = moreodeget(moreoptions,'FSA',false,'fast');
FSAerror = moreodeget(moreoptions,'FSAerror',false,'fast');
ys = moreodeget(moreoptions,'ys0',[],'fast');
ysp = moreodeget(moreoptions,'ysp0',[],'fast');
if isempty(ys) || isempty(ysp)
  FSA = false;
end
interpFcn = moreodeget(moreoptions,'interpFcn',[],'fast');
haveInterpFcn = ~isempty(interpFcn);
%usage: initialize if flag = 'init', while all the other arguments are empty. At the end of each time step taken by the solver, the input to interpFcn is flag = '',info,tnew,ynew,h,dif,k,idxNonNegative, everything needed to do interpolationm the output of interpFcn is info. Note that info is passed from one time step to another and is the output of myode15s varargout. When successfully finalizing, the flag is 'done', when the solver fails, the flag is 'fail', both with the same inputs as before.
%note that if haveInterpFcn, the output will be the output of interpFcn when flag is done, no t, or y or sol wil be outputted!

neq = length(y0);
ninst = size(ys,2);
if isempty(Mt)
  Mt = speye(neq);
end

threshold = atol / rtol;
if normcontrol
  normy = norm(y0);
  if FSA
    normys = vecnorm(ys);
  end
end

t = t0;
y = y0;

vectorJ = all(size(J)==size(y));
if vectorJ
  yp = fnlin(t,y) + J.*y;
  Mtdiag = full(diag(Mt));
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
G3(1,1,:) = G; % for FSA
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
abshFSA = absh; %record the absh and k at previous FSA stepping
kFSA = k;

dif = zeros(neq,maxk+2);
dif(:,1) = h * yp;
if FSA
  difs = zeros(neq,ninst,maxk+2);
  difs(:,:,1) = h * ysp;
end

hinvGak = h * invGa(k);
nconhk = 0;                             % steps taken with current h and k

% Allocate memory if we're generating output.
nout = 0;
tout = []; yout = [];
ysout = [];
if nargout > 0
  if output_sol
    chunk = min(max(100,50*refine), refine+floor((2^11)/neq));
    tout = zeros(1,chunk);
    yout = zeros(neq,chunk);
    kvec = zeros(1,chunk);
    dif3d = zeros(neq,maxk+2,chunk);
    if FSA
      ysout = zeros(neqsout,ninst,chunk);
      difs4d = zeros(neqsout,ninst,maxk+2,chunk);
    end
  else
    if ntspan > 2                         % output only at tspan points
      tout = zeros(1,ntspan);
      yout = zeros(neq,ntspan);
      if FSA
        ysout = zeros(neqsout,ninst,ntspan);
      end
    else                                  % alloc in chunks
      chunk = min(max(100,50*refine), refine+floor((2^13)/neq));
      tout = zeros(1,chunk);
      yout = zeros(neq,chunk);
      if FSA
        ysout = zeros(neqsout,ninst,chunk);
      end
    end
  end
  nout = 1;
  tout(nout) = t;
  yout(:,nout) = y;
  if FSA
    ysout(:,:,nout) = ys;
  end
end

if haveInterpFcn
  interp_info = feval(interpFcn,'init',[],[],[],[],[],[],[]);
end

% THE MAIN LOOP

done = false;
at_hmin = false;
while ~done

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

  % Stretch the step if within 10% of tfinal-t.
  if 1.1*absh >= abs(tfinal - t)
    h = tfinal - t;
    absh = abs(h);
    done = true;
  end

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
    if done
      tnew = tfinal;   % Hit end point exactly.
    end
    h = tnew - t;      % Purify h.
    pred = y + sum(dif(:,K),2);
    ynew = pred;

    if normcontrol
      normynew = norm(ynew);
      invwt = 1 / max(max(normy,normynew),threshold);
    else
      invwt = 1 ./ max(max(abs(y),abs(ynew)),threshold);
    end

    rhs = Mt*(ynew-psi) + hinvGak*fex;
    if vectorJ
      ynew2 = rhs ./ (Mtdiag - hinvGak*J);
    else
      ynew2 = pencil(rhs,hinvGak);
    end

    difkp1 = ynew2-ynew;
    ynew = ynew2;

    % difkp1 is now the backward difference of ynew of order k+1.
    if normcontrol
      err = (norm(difkp1) * invwt) * erconst(k);
    else
      err = norm(difkp1 .* invwt,inf) * erconst(k);
    end

    fail = (err > rtol);

    if ~fail && FSA %successful, move on to FSA
      if abshFSA ~= absh || kFSA ~= k
        difRU = cumprod((kI - 1 - kJ*(absh/abshFSA)) ./ kI) * difU;
        difRU4d = permute(difRU(K,K),[3,4,1,2]);
        difs(:,:,K) = squeeze(sum(difs(:,:,K) .* difRU4d, 3));
        abshFSA = absh;
        kFSA = k;
      end
      %time step is already known
      %the following is fixed since time step is fixed and so is difs
      psis = sum(difs(:,:,K) .* G3(1,1,K),3) * invGa(k);
      preds = ys + sum(difs(:,:,K),3);
      ynews = preds;
      if normcontrol
        normynews = vecnorm(ynews);
        invwts = 1 ./ max(max(normys,normynews),threshold);
      else
        invwts = 1 ./ max(max(abs(ys),abs(ynews)),threshold);
      end

      dfexdy = feval(Jex,t,y);
      sensval= feval(exsensFcn,t,y);
      fs = dfexdy*ys + (dJdp*ynew + sensval);
      rhss = Mt*(ynews-psis) + hinvGak*fs;
      if vectorJ
        ynews2 = rhss ./ (Mtdiag - hinvGak*J);
      else
        ynews2 = pencil(rhss,hinvGak);
      end
      difkp1s = ynews2-ynews;
      ynews = ynews2;

      if FSAerror
        if normcontrol
          errs = (vecnorm(difkp1s) .* invwts) * erconst(k);
        else
          errs = vecnorm(difkp1s .* invwts,inf) * erconst(k);
        end
        err = max(err,max(errs));
        if err > rtol
          fail = true;
        end
      end
    end

    if fail
      if absh <= hmin
        warning(message('MATLAB:ode15s:IntegrationTolNotMet', sprintf( '%e', t ), sprintf( '%e', hmin )));
        if haveInterpFcn
          interp_info = feval(interpFcn,'fail',interp_info,tnew,ynew,h,dif,k,idxNonNegative);
          varargout = interp_info;
          return;
        end
        solver_output = odefinalize_ez(sol,nout,tout,yout,kvec,dif3d,difs4d,idxNonNegative,ysout);
        if nargout > 0
          varargout = solver_output;
        end
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

  if output_sol
    nout = nout + 1;
    if nout > length(tout)
      tout = [tout, zeros(1,chunk)];  % requires chunk >= refine
      yout = [yout, zeros(neq,chunk)];
      kvec = [kvec, zeros(1,chunk)];
      dif3d = cat(3,dif3d, zeros(neq,maxk+2,chunk));
      if FSA
        ysout = cat(3,ysout, zeros(neqsout,ninst,chunk));
        difs4d = cat(4,difs4d, zeros(neqsout,ninst,maxk+2,chunk));
      end
    end
    tout(nout) = tnew;
    yout(:,nout) = ynew;
    kvec(nout) = k;
    dif3d(:,:,nout) = dif;
    if FSA
      ysout(:,:,nout) = ynews;
      difs4d(:,:,:,nout) = difs;
    end
  end

  if output_ty || haveOutputFcn
    switch outputAt
     case 'SolverSteps'        % computed points, no refinement
      nout_new = 1;
      tout_new = tnew;
      yout_new = ynew;
      if FSA
        ysout_new = ynews;
      end
     case 'RefinedSteps'       % computed points, with refinement
      tref = t + (tnew-t)*S;
      nout_new = refine;
      tout_new = [tref, tnew];
      yout_new = [ntrp15s(tref,[],[],tnew,ynew,h,dif,k,idxNonNegative), ynew];
      if FSA
        ysout_new = cat(3,ntrp15svec(tref,[],[],tnew,ynews,h,difs,k,idxNonNegative), ynews);
      end
     case 'RequestedPoints'    % output only at tspan points
      nout_new =  0;
      tout_new = [];
      yout_new = [];
      ysout_new = [];
      while next <= ntspan
        if tdir * (tnew - tspan(next)) < 0
          break;
        end
        nout_new = nout_new + 1;
        tout_new = [tout_new, tspan(next)];
        if tspan(next) == tnew
          yout_new = [yout_new, ynew];
          if FSA
            ysout_new = cat(3, ysout_new, ynews);
          end
        else
          yout_new = [yout_new, ntrp15s(tspan(next),[],[],tnew,ynew,h,dif,k,...
              idxNonNegative)];
            if FSA
              ysout_new = cat(3, ysout_new, ntrp15svec(tspan(next),[],[],tnew,ynews,h,difs,k,...
                  idxNonNegative));
            end
        end
        next = next + 1;
      end
    end

    if nout_new > 0
      if output_ty
        oldnout = nout;
        nout = nout + nout_new;
        if nout > length(tout)
          tout = [tout, zeros(1,chunk)];  % requires chunk >= refine
          yout = [yout, zeros(neq,chunk)];
          if FSA
            ysout = cat(3, ysout, zeros(neqsout,ninst,chunk));
          end
        end
        idx = oldnout+1:nout;
        tout(idx) = tout_new;
        yout(:,idx) = yout_new;
        if FSA
          ysout(:,:,idx) = ysout_new;
        end
      end
    end
  end

  if haveInterpFcn
    interp_info = feval(interpFcn,'',interp_info,tnew,ynew,h,dif,k,idxNonNegative);
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
  if FSA
    ys = ynews;
  end
  if normcontrol
    normy = normynew;
    if FSA
      normys = normynews;
    end
  end
end

if haveInterpFcn
  interp_info = feval(interpFcn,'done',interp_info,tnew,ynew,h,dif,k,idxNonNegative);
  varargout = interp_info;
  return;
end
solver_output = odefinalize_ez(sol,nout,tout,yout,kvec,dif3d,difs4d,idxNonNegative,ysout);
if nargout > 0
  varargout = solver_output;
end
end


function solver_output = odefinalize_ez(sol,nout,tout,yout,kvec,dif3d,difs4d,idxNonNegative,ysout)
  solver_output = {};
  if (nout > 0) % produce output
    if isempty(sol) % output [t,y,...]
      solver_output{1} = tout(1:nout).';
      solver_output{2} = yout(:,1:nout);
      if ~isempty(ysout)
        solver_output{3} = permute(ysout(:,:,1:nout),[3,1,2]);
      end
    else % output sol
      % Add remaining fields
      sol.x = tout(1:nout);
      sol.y = yout(:,1:nout);
      sol.idata.kvec = kvec(1:nout);
      maxkvec = max(sol.idata.kvec);
      sol.idata.dif3d = dif3d(:,1:maxkvec+2,1:nout);
      if ~isempty(difs4d)
        sol.idata.difs4d = difs4d(:,:,1:maxkvec+2,1:nout);
      end
      sol.idata.idxNonNegative = idxNonNegative;
      solver_output{1} = sol;
    end
  end
end
