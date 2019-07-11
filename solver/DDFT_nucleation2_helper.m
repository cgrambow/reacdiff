addpath('../../CHACR/GIP');
save_history = true;
resultpath = [largedatapath,'DDFT_nucleation2.mat'];

options = optimoptions('fminunc','OutputFcn',@(x,optimvalues,state) save_opt_history(x,optimvalues,state,resultpath));
x_opt = IP_DDFT(tdata,ydata,params,[21,21],'k',options,x_opt);





