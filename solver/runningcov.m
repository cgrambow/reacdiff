function [sigma,runningdyad,runningsum] = runningcov(x,runningdyad,runningsum)
  %x can be a row vector or a matrix whose columns represent instances
  %The first output sigma is the accumulative covariance
  persistent N
  if nargin<3
    runningdyad = x'*x;
    runningsum = sum(x,1);
    N = size(x,1);
  else
    runningdyad = runningdyad + x'*x;
    runningsum = runningsum + sum(x,1);
    N = N+size(x,1);
  end
  sigma = (runningdyad - runningsum'*runningsum/N)/(N-1);
end
