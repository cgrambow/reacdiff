function history_production(resultpath,ind,modelfunc,arg,tdata,ydata,params,kernelSize,Cspace,varargin)
%modelfunc and arg should both be struct with fieldnames corresponding to the parameters to be plotted
%note that ind is the indices of the optimization steps to be plotted. ind can also be 'end', in order to quickly visualize the final result.
%Use name value pair;
%'label'=true (default) to label the iteration number somewhere according to 'labelPosition'(= 'best' by default) for the first figure and to the left of each row for the second figure
%'labelxpos', coordinate of the x position of the label
%'labelHorizontalAlignment', 'left' (default), 'center','right'
%'ExchangeCurrentFieldColorScale',
%FrameIndex: selectively plot frames of c field by their index. ':' by default, meaning plotting all frames.
%Orientation: horizontal (default) arranges iterations (specified by ind) horizontally and different functions vertically; vertical does the other way
%yyaxisLim: each row corresponds to the ylim of each curve
%use CtruthSubplot as a vector to specify where to put the Ctruth
%If there is more than one functions to plot, modelfunc should be a struct, whose field names match those in the meta. If there is only one function to plot, modelfunc can either be a struct or simply a function handle
%the result file can also contain y, the solution to the PDE at each step, in which case we don't need to recompute
%IP_DDFT_arg, varargin for IP_DDFT
addpath('../../CHACR')
addpath('../../CHACR/IP')
ps = inputParser;
addParameter(ps,'label',true);
addParameter(ps,'labelPosition','top');
addParameter(ps,'labelxpos',0.1);
addParameter(ps,'labelHorizontalAlignment','left');
addParameter(ps,'ExchangeCurrentFieldColorScale',[]);
addParameter(ps,'FrameIndex',':');
addParameter(ps,'Orientation','horizontal');
addParameter(ps,'yyaxisLim',[]);
addParameter(ps,'xlim',[]);
addParameter(ps,'timeLabel',true);
addParameter(ps,'FontSize',12);
addParameter(ps,'legend',[]);
addParameter(ps,'scale',[]);
addParameter(ps,'CtruthSubplot',[]);
addParameter(ps,'IP_DDFT_arg',{});
addParameter(ps,'k0',1); %used to scale k
addParameter(ps,'dmu_at_0',1); %when meta has both mu and C, rescale so that dmu/dx at 0 has this value
addParameter(ps,'mu_at_0',0); %offset mu so that mu at 0 has this value
parse(ps,varargin{:});
ps = ps.Results;

varload = load(resultpath);
history = varload.history;
if isempty(ind)
  ind = 1:size(history,1);
elseif isequal(ind,'end')
  ind = size(history,1);
end


numIter = length(ind);
if isempty(ps.scale)
  ps.scale = false(1,numIter);
end

frameindex = ps.FrameIndex;
if isequal(frameindex,':')
  column = size(ydata,1);
  frameindex = 1:column;
else
  column = length(frameindex);
end
rowtotal = numIter+1;
columntotal = column;
frameindex_model = frameindex;
frameindex_model(frameindex==1) = []; %remove the initial condition

% 'color',[0.8500,0.3250,0.0980]
stparg = {0.05,[0.05,0.08],0.05};
figure;
clim = [min(min(ydata(frameindex,:))),max(max(ydata(frameindex,:)))];
h = visualize([],[],[],ydata(frameindex,:),'c',false,'ImageSize',params.N,'caxis',clim,'GridSize',[1,NaN],'OuterGridSize',[rowtotal,1],'OuterSubplot',[1,1],'ColumnTotal',columntotal,'StarterInd',0,'subtightplot',stparg);
if ps.timeLabel
  for j = 1:length(h)
    title(h(j),['t = ',num2str(tdata(frameindex(j)),2)]);
  end
  axes(h(1));
  text(0.1,1.5,'Data','Units','normalized','HorizontalAlignment',ps.labelHorizontalAlignment);
end
for i = 1:numIter
  if isfield(varload,'y')
    y = varload.y;
    visualize([],[],[],y{ind(i)}(frameindex_model,:),'c',false,'ImageSize',params.N,'caxis',clim,'GridSize',[1,NaN],'OuterGridSize',[rowtotal,1],'OuterSubplot',[i+1,1],'ColumnTotal',columntotal,'StarterInd',i*columntotal+1,'subtightplot',stparg);
  else
    [yhistory,~,~,pp] = IP_DDFT(tdata,ydata,params,kernelSize,Cspace,[],history(ind(i),:),'mode','eval',ps.IP_DDFT_arg{:});
    visualize([],[],[],yhistory(frameindex_model,:),'c',false,'ImageSize',params.N,'caxis',clim,'GridSize',[1,NaN],'OuterGridSize',[rowtotal,1],'OuterSubplot',[i+1,1],'ColumnTotal',columntotal,'StarterInd',i*columntotal+1,'subtightplot',stparg);
  end
  meta = pp.meta;
  names = fieldnames(meta);
  namesflag = ismember({'mu','C'},names);
  numFunc = numel(names);
  subtightplot(rowtotal,columntotal,i*columntotal+1,stparg{:});
  if all(namesflag)
    %calculate current gradient of mu at x=0
    grad0 = customizeFunGrad(pp.params,'mu','grad',0);
    %gradient of the first order basis function of mu
    [~,grad1] = feval(pp.params.mu.func,0,1);
    %set the gradient of mu at x=0 to be 1. The offset is
    offset = (ps.dmu_at_0-grad0)/grad1;
    pp.params.mu.params(1) = pp.params.mu.params(1) + offset;
  else
    offset = 0;
  end
  for j=1:numFunc
    name = names{j};
    if ismember(name,{'D','extdata'})
      continue;
    end
    if isequal(name,'C') && ismember(Cspace,{'k','real'})
      imagesc(pp.params.C-offset);
      caxis([0,1]);
      ax = gca;
      axis(ax,'image');
      ax.XTick = [];
      ax.YTick = [];
      ax.Box = 'off';
    else
      if isequal(name,'C') && ~isempty(regexp(Cspace,'isotropic*'))
        customfunc = pp.params.Cfunc.func;
        customparams = pp.params.Cfunc.params;
        yoffset = offset;
      elseif isequal(name,'mu')
        customfunc = pp.params.(name).func;
        customparams = pp.params.(name).params;
        yoffset = - customfunc(0,customparams);
      end
      if numFunc>1
        if j==1
          yyaxis left
        elseif j==2
          yyaxis right
        end
      end
      argj = arg.(name);
      if isequal(name,'C') && ~isempty(regexp(Cspace,'isotropic*'))
        xscale = ps.k0;
      else
        xscale = 1;
      end
      plot(argj,modelfunc.(name)(argj*xscale),'--','LineWidth',2);
      ax = gca;
      ylimtemp = ax.YLim;
      hold on;
      plot(argj,customfunc(argj*xscale,customparams)+yoffset,'LineWidth',2);
      if isempty(ps.yyaxisLim)
        ax.YLim = ylimtemp;
      else
        ax.YLim = ps.yyaxisLim(j,:);
      end
      if ~isempty(ps.xlim)
        ax.XLim = ps.xlim;
      end
      axis square
      if i~=numIter
        ax.XTick = [];
      else
        if namesflag(1) && ~namesflag(2)
          xl = '\psi';
        elseif ~namesflag(1) && namesflag(2)
          xl = 'k/k_0';
        elseif all(namesflag)
          xl = '\psi or k/k_0';
        end
        xlabel(xl);
        if ~isempty(ps.legend)
          legend(ps.legend{:});
          legend('boxoff');
        end
      end
    end
    if ps.label && j==1
      text(0.1,1.2,['Iter. ',num2str(ind(i)-1)],'Units','normalized','HorizontalAlignment',ps.labelHorizontalAlignment);
    end
  end
end
if isfield(meta,'C') && ismember(Cspace,{'k','real'}) && ~isempty(ps.CtruthSubplot)
  axC = subtightplot(rowtotal,columntotal,columntotal*(ps.CtruthSubplot(1)-1)+ps.CtruthSubplot(2),stparg{:});
  CC = params.C;
  halfSize = (kernelSize-1)/2;
  CC((2+halfSize(1)):(end-halfSize(1)),:)=[];
  CC(:,(2+halfSize(2)):(end-halfSize(2)))=[];
  imagesc(fftshift(CC));
  axis(axC,'image');
  axC.XTick = [];
  axC.YTick = [];
  axC.Box = 'off';
  title('C_2 (truth)')
  colormap(gca,'default')
end
set(findall(gcf,'-property','FontName'),'FontName','Arial');
set(findall(gcf,'-property','FontWeight'),'FontWeight','normal');
set(findall(gcf,'-property','FontSize'),'FontSize',ps.FontSize);

end

function hOut = TextLocation(textString,varargin)

l = legend(textString,varargin{:});
t = annotation('textbox');
t.String = textString;
t.Position = l.Position;
delete(l);
% t.LineStyle = 'None';

if nargout
    hOut = t;
end
end
