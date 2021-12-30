function history_movie(resultpath,vid,ind,modelfunc,arg,tdata,ydata,params,kernelSize,Cspace, varargin)
%based on history_movie in CHACR/IP and history_production in RD/solver , produce a movie.
%must create a figure outside and a movie writer outside and pass the video object (vid)
%below are instructions from history_production, which may or may not be relevant here
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
%columntotal and rowtotal, overwrite the total number of columns and rows
%If there is more than one functions to plot, modelfunc should be a struct, whose field names match those in the meta. If there is only one function to plot, modelfunc can either be a struct or simply a function handle
%the result file can also contain y, the solution to the PDE at each step, in which case we don't need to recompute
%vararagout{1} is the all the subplots handles in 2D array form
ps = inputParser;
addParameter(ps,'funclabel',false); %label for functions
addParameter(ps,'funcxlabel',true); %xlabel for functions
addParameter(ps,'label',true); %label for iteration
addParameter(ps,'labelPosition','top');
addParameter(ps,'labelpos',[0,1.2]);
addParameter(ps,'labelHorizontalAlignment','left');
addParameter(ps,'ExchangeCurrentFieldColorScale',[]);
addParameter(ps,'FrameIndex',':');
addParameter(ps,'Orientation','horizontal');
addParameter(ps,'yyaxisLim',[]);
addParameter(ps,'timeLabel',true); %time is shifted such that t for the first frame is 0
addParameter(ps,'timeLabelManual',[]); %replace the default time label above, must be a cell of char arrays
addParameter(ps,'FontSize',12);
addParameter(ps,'scale',[]);
addParameter(ps,'targetLineStyle','--');
addParameter(ps,'title',[]); %left aligned, on top of the first data row
addParameter(ps,'stparg',{0.05,[0.1,0.08],0.05});  %argument for subtightplot
addParameter(ps,'visualizeArgs',{}); %additional arguments for visualize
addParameter(ps,'color',linspecer(2)); %color for lines
addParameter(ps,'save',true); %whether to save the ymodel result in resultpath
addParameter(ps,'use_saved',true); %whether to use saved ymodel result in resultpath
addParameter(ps,'darkmode',true); %generate darkmode movies
addParameter(ps,'offset',true); %if both C and mu are optimized. Whether to offset C with a constant term and offset mu with a the linear term
addParameter(ps,'muderiv',false); %plot mu'(c) instead of mu
addParameter(ps,'component',[]); %specify what functions to plot and in what order, only two allowed
addParameter(ps,'IP_DDFT_arg',{});
addParameter(ps,'k0',1); %used to scale k
addParameter(ps,'dmu_at_0',1); %when meta has both mu and C, rescale so that dmu/dx at 0 has this value
addParameter(ps,'mu_at_0',0); %offset mu so that mu at 0 has this value

parse(ps,varargin{:});
ps = ps.Results;

varload = matfile(resultpath);
history = varload.history;
has_y = ~isempty(who(varload,'y'));

if isempty(ind)
  ind = 1:size(history,1);
end

%plot target and fitted function
if isempty(ps.component)
  names = fieldnames(modelfunc);
  names = names(1:2);
  names = flip(sort(names));
else
  names = ps.component;
end
numFunc = numel(names);
numIter = length(ind);


frameindex = ps.FrameIndex;
if isequal(frameindex,':')
  column = size(ydata,1);
  frameindex = 1:column;
else
  column = length(frameindex);
end
columntotal = column + 1;
rowtotal = 2;

frameindex_model = frameindex;

stparg = ps.stparg;

clim = [min(min(ydata(frameindex,:))),max(max(ydata(frameindex,:)))];

hdata = visualize([],[],[],ydata(frameindex,:),'c',false,'ImageSize',params.N,'caxis',clim,'GridSize',[1,NaN],'OuterGridSize',[rowtotal,1],'OuterSubplot',[1,1],'ColumnTotal',columntotal,'StarterInd',1,'subtightplot',stparg,ps.visualizeArgs{:});
data_cm = hdata(1).Colormap;

if has_y && ps.use_saved
  y = varload.y;
else
  y = {};
end

open(vid);
for i = 1:numIter
  if has_y && ps.use_saved
    ymodel = y{ind(i)};
    [~,~,~,pp] = IP_DDFT(tdata,ydata,params,kernelSize,Cspace,[],history(ind(i),:),'mode','pp',ps.IP_DDFT_arg{:});
  else
    [ymodel,~,~,pp] = IP_DDFT(tdata,ydata,params,kernelSize,Cspace,[],history(ind(i),:),'mode','eval',ps.IP_DDFT_arg{:});
    if ps.save
      y{ind(i)} = ymodel;
    end
  end
  hmodel = visualize([],[],[],ymodel(frameindex_model,:),'c',false,'ImageSize',params.N,'caxis',clim,'GridSize',[1,NaN],'OuterGridSize',[rowtotal,1],'OuterSubplot',[1,1],'ColumnTotal',columntotal,'StarterInd',columntotal+1,'subtightplot',stparg,ps.visualizeArgs{:});
  if all(ismember({'mu','C'},names)) && ps.offset
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
    hfunc(j) = subtightplot(rowtotal,columntotal,(j-1)*columntotal+1,stparg{:});
    name = names{j};
    if isequal(name,'C') && ~isempty(regexp(Cspace,'isotropic*'))
      customfunc = pp.params.Cfunc.func;
      customparams = pp.params.Cfunc.params;
      yoffset = offset;
    elseif isequal(name,'mu')
      customfunc = pp.params.(name).func;
      customparams = pp.params.(name).params;
      yoffset = - customfunc(0,customparams);
    end
    argj = arg.(name);
    if isequal(name,'C') && ~isempty(regexp(Cspace,'isotropic*'))
      xscale = ps.k0;
    else
      xscale = 1;
    end
    if isequal(name,'mu') && ps.muderiv
      [~,yarg] = modelfunc.(name)(argj*xscale);
    else
      yarg = modelfunc.(name)(argj*xscale);
    end
    plot(argj,yarg,ps.targetLineStyle,'LineWidth',2,'Color',ps.color(j,:));
    ax = gca;
    ylimtemp = ax.YLim;
    hold on;
    if isequal(name,'mu') && ps.muderiv
      [~,yarg] = customfunc(argj*xscale,customparams);
    else
      yarg = customfunc(argj*xscale,customparams)+yoffset;
    end
    plot(argj,yarg,'-','LineWidth',2,'Color',ps.color(j,:));
    if isempty(ps.yyaxisLim)
      ax.YLim = ylimtemp;
    elseif all(isfinite(ps.yyaxisLim(j,:)))
      ax.YLim = ps.yyaxisLim(j,:);
    end
    axis square
    switch name
    case 'mu'
      ax.XTick = [];
      clb = colorbar('southoutside','Position',[ax.Position(1),ax.Position(2)+0.04,ax.Position(3),0.03],'LineWidth',1);
      if ps.darkmode
        clb.Color = [1,1,1];
      end
    case 'C'
      if ps.funcxlabel
        xlabel('k/k_0');
      end
    end
    if ps.darkmode
      ax.Color = [0,0,0];
      ax.XColor = [1,1,1];
      ax.YColor = [1,1,1];
    end
    ax.Colormap = data_cm;
    if ps.funclabel
      switch name
      case 'mu'
        if ps.muderiv
          lab = "\mu_h'(c)";
        else
          lab = '\mu_h(c)';
        end
      case 'C'
          lab = 'C_2(k)';
      end
      tx = title(lab,'FontWeight','normal');
      if ps.darkmode
        tx.Color = [1,1,1];
      end
    end
  end
  if ps.label
    tx = text(hfunc(1),ps.labelpos(1),ps.labelpos(2),['Iteration ',num2str(ind(i)-1)],'Units','normalized','HorizontalAlignment',ps.labelHorizontalAlignment);
    if ps.darkmode
      tx.Color = [1,1,1];
    end
  end
  set(findall(gcf,'-property','FontName'),'FontName','Arial');
  set(findall(gcf,'-property','FontWeight'),'FontWeight','normal');
  set(findall(gcf,'-property','FontSize'),'FontSize',ps.FontSize);
  F = print('-RGBImage');
  writeVideo(vid,F);
  delete(hfunc);
  delete(hmodel);
  if ps.label
    delete(tx);
  end
end
close(vid);

if ps.save
  varload.Properties.Writable = true;
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
