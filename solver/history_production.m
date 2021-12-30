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
addParameter(ps,'xlabel','on');
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
addParameter(ps,'targetLineStyle','-.');
addParameter(ps,'offset',true); %if both C and mu are optimized. Whether to offset C with a constant term and offset mu with a the linear term
addParameter(ps,'muderiv',false); %plot mu'(c) instead of mu
addParameter(ps,'showModelSolution',true); %plot the model solution at each iteration
addParameter(ps,'stparg',{0.05,[0.05,0.08],0.05});  %argument for subtightplot
parse(ps,varargin{:});
ps = ps.Results;

varload = matfile(resultpath);
history = varload.history;
savey = isempty(who(varload,'y'));
if savey
  varload.Properties.Writable = true;
end
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
if ps.showModelSolution
  rowtotal = numIter+1;
  columntotal = column;
  frameindex_model = frameindex;
  frameindex_model(frameindex==1) = []; %remove the initial condition
else
  rowtotal = ceil(numIter/column) + 1;
  columntotal = column;
end

% 'color',[0.8500,0.3250,0.0980]
stparg = ps.stparg;
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

if savey
  y = {};
else
  y = varload.y;
end

for i = 1:numIter
  if ~ps.showModelSolution || (ps.showModelSolution && ~savey)
    [~,~,~,pp] = IP_DDFT(tdata,ydata,params,kernelSize,Cspace,[],history(ind(i),:),'mode','pp',ps.IP_DDFT_arg{:});
  end
  if ps.showModelSolution
    if ~savey
      y = varload.y;
      visualize([],[],[],y{ind(i)}(frameindex_model,:),'c',false,'ImageSize',params.N,'caxis',clim,'GridSize',[1,NaN],'OuterGridSize',[rowtotal,1],'OuterSubplot',[i+1,1],'ColumnTotal',columntotal,'StarterInd',i*columntotal+1,'subtightplot',stparg);
    else
      [yhistory,~,~,pp] = IP_DDFT(tdata,ydata,params,kernelSize,Cspace,[],history(ind(i),:),'mode','eval',ps.IP_DDFT_arg{:});
      y{ind(i)} = yhistory;
      visualize([],[],[],yhistory(frameindex_model,:),'c',false,'ImageSize',params.N,'caxis',clim,'GridSize',[1,NaN],'OuterGridSize',[rowtotal,1],'OuterSubplot',[i+1,1],'ColumnTotal',columntotal,'StarterInd',i*columntotal+1,'subtightplot',stparg);
    end
    ax = subtightplot(rowtotal,columntotal,i*columntotal+1,stparg{:});
  else
    ax = subtightplot(rowtotal,columntotal,columntotal+i,stparg{:});
  end


  if (ps.showModelSolution && i==numIter) || (~ps.showModelSolution && i==1)
    ps.xlabel = 'on';
  else
    ps.xlabel = [];
  end

  history_func(ind(i),modelfunc,arg,Cspace,pp,ps);

end

if ismember(Cspace,{'k','real'}) && ~isempty(ps.CtruthSubplot)
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

if savey
  varload.y = y;
end

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
