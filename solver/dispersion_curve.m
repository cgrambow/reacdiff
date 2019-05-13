A1 = [0.08,-0.08;0.1,0];
A2 = -[0.03,0.08];
A = A1+diag(A2);
D = [0.02,0.5];

k = linspace(0,2,1000);
a = 1;
b = (-trace(A)+sum(D)*k.^2);
c = (prod(D)*k.^4-(A(1,1)*D(2)+A(2,2)*D(1))*k.^2+det(A));
lambda = (-b+[1;-1]*sqrt(b.^2-4*a*c))./(2*a);
figure;
plot(k,max(real(lambda)),'LineWidth',2);
axis([0,2,-0.05,0.02]);
xlabel('|k|');
ylabel('max Re \lambda')
print(gcf,'C:\Users\zhbkl\Dropbox (MIT)\MIT Courses\2.168\2.168 Project\Final\dispersion','-dpng','-r400');
