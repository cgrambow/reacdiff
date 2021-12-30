D = [0.02,0.5];
A1 = [0.08,-0.08;0.1,0];
A2 = -[0.03,0.08];
mat_ss = matfile('/home/hbozhao/Dropbox (MIT)/2.168 Project/Data/turing_ss');
A1_ss = mean(mat_ss.A1,1);
A1lb = [-0.5,A1_ss(3);-0.5,A1_ss(4)];
A1ub = [1.5,A1_ss(3);1.5,A1_ss(4)];
Anorm = norm(A1+diag(A2),inf);

ind = 0;
while ind < 10000
  A1new = A1lb + (A1ub-A1lb).*rand(2);
  Anew = A1new+diag(A2);
  if LSA(Anew,D) && (norm(Anew,inf)<5*Anorm)
    ind = ind+1;
    Alist(ind,:) = reshape(A1new,1,[]);
  end
end
