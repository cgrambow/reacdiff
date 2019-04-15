function varargout = k2real(y,N,y00,varargin)
  %convert from k space to real space
  %y comes in as a column vector, each column represent different snapshots Nt
  %N is an array for the system size
  %y00 is the k=0 component if missing in y, and must have the size of (1-Nt)
  %mode: plot (default): show static snapshots; movie; value: calculate the y value in real space, returned in the same format
  %caxis: 'auto' (default): each snapshot has its own automatic caxis; 'max': the caxis that accomondates the maximum and minimum of all the snapshots (only works for mode='plot'); 'first': all snapshots follow the min and max of the first snapshot; 'last': all snapshots follow the min and max of the last snapshot; a scalar, all snapshots follow the min and max of the snapshot numbered by the scalar; otherwise, all snapshots' clim specified by what's given.
  Nt = size(y,2);
  defaultcolumn = max(factor(Nt));
  defaultrow = Nt/defaultcolumn;
  if length(y) == prod(N)-1
    if nargin<3 || isempty(y00)
      y00 = zeros(1,Nt);
    elseif isscalar(y00)
      y00 = y00*ones(1,Nt);
    end
    y = [y00;y];
  end
  ps = inputParser;
  addParameter(ps,'mode','plot');
  addParameter(ps,'caxis','auto');
  addParameter(ps,'GridSize',[defaultrow,defaultcolumn]);
  ps.KeepUnmatched = true;
  parse(ps,varargin{:});
  params = ps.Results;
  row = params.GridSize(1);
  column = params.GridSize(2);

  if isscalar(params.caxis)
    yreal = real(ifftn(reshape(y(:,params.caxis),N)));
    yreal = yreal(:);
    clim = [min(yreal),max(yreal)];
  else
    switch params.caxis
    case 'max'
      clim = 'auto';
    case 'first'
      yreal = real(ifftn(reshape(y(:,1),N)));
      yreal = yreal(:);
      clim = [min(yreal),max(yreal)];
    case 'last'
      yreal = real(ifftn(reshape(y(:,end),N)));
      yreal = yreal(:);
      clim = [min(yreal),max(yreal)];
    otherwise
      clim = params.caxis;
    end
  end

  switch params.mode
  case 'plot'
    vis = 1;
  case 'movie'
    vis = 2;
    F(Nt) = struct('cdata',[],'colormap',[]);
  case 'value'
    vis = 0;
  end
  for i=1:Nt
    yreal = real(ifftn(reshape(y(:,i),N)));
    if vis == 1
      h(i) = subplot(row,column,i);
      imagesc(yreal);
      axis(h(i),'equal','tight');
      colormap(h(i),'gray');
      caxis(h(i),clim);
    elseif vis == 2
      h = imagesc(yreal);
      axis(gca,'equal','tight');
      colormap(gca,'gray');
      caxis(gca,clim);
      F(i) = getframe(gcf);
      delete(h);
    else
      y(:,i) = yreal(:);
    end
    ax = gca;
    ax.XTick=[]; ax.YTick=[];
  end
  if isequal(params.caxis,'max') && vis == 1
    for i=1:Nt
      climi = h(i).CLim;
      if i==1
        clim = climi;
      else
        clim(1) = min(clim(1),climi(1));
        clim(2) = max(clim(2),climi(2));
      end
    end
    for i=1:Nt
      caxis(h(i),clim);
    end
  end
  if vis == 1
    varargout = {h};
  elseif vis == 2
    varargout = {F};
  else
    varargout = {y};
  end
end
