function history_func(ind,modelfunc,arg,Cspace,pp,property,varargin)
  %show the functional forms of history
  %pp is the post-processing output from IP_DDFT
  %property can be provided directly, if not, through varargin
  %ind is the index
  if isempty(property)
    ps = inputParser;
    ps.KeepUnmatched = true;
    addParameter(ps,'xlabel','on');
    addParameter(ps,'dmu_at_0',1); %when meta has both mu and C, rescale so that dmu/dx at 0 has this value
    addParameter(ps,'k0',1); %used to scale k
    addParameter(ps,'yyaxisLim',[]);
    addParameter(ps,'xlim',[]);
    addParameter(ps,'legend',[]);
    addParameter(ps,'label',true);
    addParameter(ps,'labelHorizontalAlignment','left');
    addParameter(ps,'targetLineStyle','-.');
    parse(ps,varargin{:});
    ps = ps.Results;
  else
    ps = property;
  end

  meta = pp.meta;
  names = fieldnames(meta);
  namesflag = ismember({'mu','C'},names);
  numFunc = numel(names);

  if all(namesflag)
    %calculate current gradient of mu at x=0
    grad0 = customizeFunGrad(pp.params,'mu','grad',0);
    %gradient of the first order basis function of mu
    [~,grad1] = feval(pp.params.mu.func,0,1);
    %set the gradient of mu at x=0 to be 1. The offset is
    offset = (ps.dmu_at_0-grad0)/grad1;
    pp.params.mu.params(1) = pp.params.mu.params(1) + offset;

    % shift C2 such that C2(0) matches the truth
    % offset = modelfunc.C(0) - customizeFunGrad(pp.params,'Cfunc','fun',0);
    % pp.params.mu.params(1) = pp.params.mu.params(1) + offset/grad1;
  else
    offset = 0;
  end
  ax = gca;
  for j=1:numFunc
    name = names{j};
    if ismember(name,{'D','extdata'})
      continue;
    end
    if isequal(name,'C') && ismember(Cspace,{'k','real'})
      imagesc(pp.params.C-offset);
      caxis([0,1]);
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
      if numFunc==1
        ax.ColorOrderIndex = j;
      end
      plot(argj,modelfunc.(name)(argj*xscale),ps.targetLineStyle,'LineWidth',2);
      ylimtemp = ax.YLim;
      hold on;
      if numFunc==1
        ax.ColorOrderIndex = j;
      end
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
      if ~isequal(ps.xlabel,'on')
        ax.XTick = [];
      else
        if namesflag(1) && ~namesflag(2)
          xl = '\eta';
        elseif ~namesflag(1) && namesflag(2)
          xl = 'k/k_0';
        elseif all(namesflag)
          xl = '\eta or k/k_0';
        end
        xlabel(xl);
        if ~isempty(ps.legend)
          legend(ps.legend{:});
          legend('boxoff');
        end
      end
    end
    if ps.label && j==1 && ~isempty(ind)
      text(0.1,1.2,['Iter. ',num2str(ind-1)],'Units','normalized','HorizontalAlignment',ps.labelHorizontalAlignment);
    end
  end
