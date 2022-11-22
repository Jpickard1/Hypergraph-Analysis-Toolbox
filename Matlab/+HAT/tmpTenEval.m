function [jacob, polyv, polyt] = tmpTenEval(x, pCoef, s)

m = zeros(17,1);
m(1) =x(2)*x(1);
m(2) =x(1)*x(3);
m(3) =x(1)*x(4);
m(4) =x(1)*x(5);
m(5) =x(1)*x(6);
m(6) =x(2)*x(2);
m(7) =x(2)*x(3);
m(8) =x(3)*x(3);
m(9) =x(4)*x(4);
m(10) =x(5)*x(5);
m(11) =x(6)*x(6);
m(12) =pCoef(2)*x(4)+pCoef(3)*x(5);
m(13) =pCoef(5)*x(4)+pCoef(6)*x(5);
m(14) =pCoef(12)*m(9);
m(15) =pCoef(16)*m(10);
m(16) =pCoef(19)*m(11);
m(17) =pCoef(20);
polyv = zeros(6,1);
polyv(1) =x(3)*m(12)+pCoef(1)*m(6)+pCoef(4)*m(1)*x(2);
polyv(2) =x(2)*m(13)+pCoef(7)*m(8)+pCoef(8)*m(2)*x(3);
polyv(3) =x(1)*m(14)+pCoef(9)*m(7)+pCoef(10)*m(9)+pCoef(11)*x(5)*x(6);
polyv(4) =x(1)*m(15)+pCoef(13)*m(7)+pCoef(14)*x(4)*x(6)+pCoef(15)*m(10);
polyv(5) =x(1)*m(16)+pCoef(17)*x(4)*x(5)+pCoef(18)*m(11);
polyv(6) =x(2)*m(17)+pCoef(21)*x(3)+pCoef(22)*x(4)+pCoef(23)*x(5)+pCoef(24)*x(6)+pCoef(25);
jacob = zeros(6,6);
jacob(1,1) = pCoef(4)*m(6);
jacob(1,2) = pCoef(1)*2*x(2) + pCoef(4)*2*m(1);
jacob(1,3) =m(12);
jacob(1,4) = pCoef(2)*x(3);
jacob(1,5) = pCoef(3)*x(3);
jacob(2,1) = pCoef(8)*m(8);
jacob(2,2) =m(13);
jacob(2,3) = pCoef(7)*2*x(3) + pCoef(8)*2*m(2);
jacob(2,4) = pCoef(5)*x(2);
jacob(2,5) = pCoef(6)*x(2);
jacob(3,1) =m(14);
jacob(3,2) = pCoef(9)*x(3);
jacob(3,3) = pCoef(9)*x(2);
jacob(3,4) = pCoef(10)*2*x(4) + pCoef(12)*2*m(3);
jacob(3,5) = pCoef(11)*x(6);
jacob(3,6) = pCoef(11)*x(5);
jacob(4,1) =m(15);
jacob(4,2) = pCoef(13)*x(3);
jacob(4,3) = pCoef(13)*x(2);
jacob(4,4) = pCoef(14)*x(6);
jacob(4,5) = pCoef(15)*2*x(5) + pCoef(16)*2*m(4);
jacob(4,6) = pCoef(14)*x(4);
jacob(5,1) =m(16);
jacob(5,4) = pCoef(17)*x(5);
jacob(5,5) = pCoef(17)*x(4);
jacob(5,6) = pCoef(18)*2*x(6) + pCoef(19)*2*m(5);
jacob(6,2) =m(17);
jacob(6,3) = pCoef(21);
jacob(6,4) = pCoef(22);
jacob(6,5) = pCoef(23);
jacob(6,6) = pCoef(24);
polyt = zeros(6,1);
if(s<-1e-10)
	[Q, DQ] = tmpEvalQDQ(x);
	polyt = exp(s)*(polyv-Q);
	polyv = Q+polyt;
	jacob = DQ+exp(s)*(jacob-DQ);
end

end

function [Q, DQ] = tmpEvalQDQ(x)

a = [ (0.395600-0.918400i);  (0.829900-0.558000i);  (0.698200+0.715900i);  (0.855500-0.517800i);  (-0.673700-0.739000i); ];
b = [(0.818000+0.575200i); (-0.178100+0.984000i); (-0.956900-0.290300i); (0.964600-0.263800i); (0.975800-0.218800i); ];
L = [(0.628200+0.778000i); (-0.881600+0.472000i); (0.863100-0.505100i); (0.262100-0.965000i); (0.967800-0.251800i); ];
C = [(0.548400+0.836200i) (0.983000-0.183700i) (0.964000-0.265900i) (-0.995800+0.091800i) (0.310700-0.950500i) ];
xm_2=x(2:end).^1;
xm_1_b=x(2:end).*xm_2-b;
ax1_L=a.*(x(1)-L);
Q=ax1_L.*xm_1_b;
Q(6) = C*x(2:6)-1;
DQ = zeros(6);
for ia=1:5
	DQ(ia, ia+1) = 2*ax1_L(ia)*xm_2(ia);
end
DQ(1:5, 1) = a.*xm_1_b;
DQ(6,2:6)=C;
end

