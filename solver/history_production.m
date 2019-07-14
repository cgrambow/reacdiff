function history_production(resultpath,ind,modelfunc,arg,meta,tdata,ydata,params,kernelSize,Cspace,varargin)
%Two figures for figure handle
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
ps = inputParser;
addParameter(ps,'label',true);
addParameter(ps,'labelPosition','top');
addParameter(ps,'labelxpos',0.1);
addParameter(ps,'labelHorizontalAlignment','left');
addParameter(ps,'ExchangeCurrentFieldColorScale',[]);
addParameter(ps,'FrameIndex',':');
addParameter(ps,'Orientation','horizontal');
addParameter(ps,'yyaxisLim',[]);
addParameter(ps,'timeLabel',true);
addParameter(ps,'FontSize',12);
addParameter(ps,'legend',[]);
addParameter(ps,'scale',[]);
addParameter(ps,'CtruthSubplot',[]);
parse(ps,varargin{:});
ps = ps.Results;

varload = load(resultpath);
history = varload.history;
history = [zeros(1,size(history,2)); history]; %hardcoded!!!
if isempty(ind)
  ind = 1:size(history,1);
elseif isequal(ind,'end')
  ind = size(history,1);
end


%plot target and fitted function
names = fieldnames(meta);
numFunc = numel(names);
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
    yhistory = IP_DDFT(tdata,ydata,params,kernelSize,Cspace,[],history(ind(i),:),'eval',true);
    visualize([],[],[],yhistory(frameindex_model,:),'c',false,'ImageSize',params.N,'caxis',clim,'GridSize',[1,NaN],'OuterGridSize',[rowtotal,1],'OuterSubplot',[i+1,1],'ColumnTotal',columntotal,'StarterInd',i*columntotal+1,'subtightplot',stparg);
  end
  subtightplot(rowtotal,columntotal,i*columntotal+1,stparg{:});
  for j=1:numFunc
    name = names{j};
    if isequal(name,'kappa')
      continue;
    end
    if isequal(name,'C')
      C = history(ind(i),:);
      C = [C,flip(C(1:end-1))];
      C = reshape(C,kernelSize);
      imagesc(C);
      caxis([0,1]);
      ax = gca;
      axis(ax,'image');
      ax.XTick = [];
      ax.YTick = [];
      ax.Box = 'off';
    else
      if j==1
        yyaxis left
      elseif j==2
        yyaxis right
      end
      plot(arg,target{j},'--','LineWidth',2);
      ax = gca;
      ylimtemp = ax.YLim;
      hold on;
      if isfield(meta,'kappa') && ps.scale(i)
        scale = exp(history(ind(i),meta.kappa.index)) / params.kappa;
        if isequal(name,'ChemicalPotential')
          scale = 1/scale;
        end
      else
        scale = 1;
      end
      plot(arg,customfunc(arg,history(ind(i),meta.(name).index))*scale,'LineWidth',2);
      if isempty(ps.yyaxisLim)
        ax.YLim = ylimtemp;
      else
        ax.YLim = ps.yyaxisLim(j,:);
      end
      axis square
    end
    if ps.label && j==1
      text(0.1,1.2,['Iter. ',num2str(ind(i)-1)],'Units','normalized','HorizontalAlignment',ps.labelHorizontalAlignment);
    end
  end
end
if isfield(meta,'C') && ~isempty(ps.CtruthSubplot)
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
