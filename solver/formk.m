function [k2,k] = formk(N,L)
%form the kx, ky, ... based on the number of points in each direction (N), k as in exp(ikx), the index goes from 0 to N-1, L is the size of the domain
%k2 is ||k|| (2-norm)
k2 = 0;
for dim = 1:length(N)
  Nmid = floor(N(dim)/2+1);
  if dim>1
    sgind = num2cell(ones(1,dim));
    sgind{end} = ':';
    k{dim}(sgind{:}) = [0:Nmid-1,Nmid-N(dim):-1]'/L(dim)*2*pi;
  else
    k{dim} = [0:Nmid-1,Nmid-N(dim):-1]'/L(dim)*2*pi;
  end
  k2 = k2 + k{dim}.^2;
end
