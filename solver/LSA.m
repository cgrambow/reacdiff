function y = LSA(A,D)
  %whether the input parameters satisfy the pattern forming condition from linear stability analysis (Page 11 of support Kondo paper)
  %only support one instance, A is a matrix, D is a vector
  y = (trace(A)<0) && (det(A)>0) && (A(1,1)*D(2)+A(2,2)*D(1)>0) && ((A(1,1)*D(2)+A(2,2)*D(1))^2 - 4*D(1)*D(2)*det(A)>0);
end
