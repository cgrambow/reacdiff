addpath('../../CHACR/GIP');
save_history = true;
resultpath = [largedatapath,'DDFT_nucleation2.mat'];
varload = load(resultpath);
x_opt = varload.history(end,:);
exitflag = 5;
options = optimoptions('fminunc','OutputFcn',@(x,optimvalues,state) save_opt_history(x,optimvalues,state,resultpath,[],true));

while (exitflag==5)
  [x_opt,~,exitflag] = IP_DDFT(tdata,ydata,params,[21,21],'k',options,x_opt);
end
