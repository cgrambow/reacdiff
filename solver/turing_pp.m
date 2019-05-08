%image post-processing
filepath = '~/Dropbox (MIT)/2.168 Project/Data/turing_ss';
mat = matfile(filepath);
ysize = size(mat,'y');
threshold = 3.5; %a separation of the bimodal distribution of the entire dataset (component 1)
%see for example:
%figure; histogram(mat.y(1:ysize(1),1:ysize(2),1:ysize(3),ysize(4),1)
feature = zeros(ysize(1),8);
index = zeros(ysize(1),1);
ind = 0;
for i=1:ysize(1)
  im = squeeze(mat.y(i,1:ysize(2),1:ysize(3),ysize(4),1));
  if ~any(im(:)>threshold)
    continue;
  end
  bw = (im>threshold);
  stats = regionprops(bw,'Area','Perimeter');
  area = [stats(:).Area];
  if ~any(area>10)
    continue;
  end
  perimeter = [stats(:).Perimeter];
  circularity = 4*pi*area./perimeter.^2;
  ind = ind + 1;
  %number of domains
  feature(ind,1) = numel(stats);
  %mean of domain area
  feature(ind,2) = mean(area);
  %std of domain area
  feature(ind,3) = std(area);
  %mean of domain perimeter
  feature(ind,4) = mean(perimeter);
  feature(ind,5) = std(perimeter);
  %min of domain circularity
  feature(ind,6) = min(circularity);
  %mean of domain circularity
  feature(ind,7) = mean(circularity);
  %std of domain circularity
  feature(ind,8) = std(circularity);
  index(ind) = i;
end
feature(ind+1:end,:) = [];
index(ind+1:end) = [];
