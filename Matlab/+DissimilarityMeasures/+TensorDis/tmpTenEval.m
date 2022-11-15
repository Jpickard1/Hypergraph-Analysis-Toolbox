function [jacob, polyv, polyt] = tmpTenEval(x, pCoef, s)

m = zeros(83,1);
m(1) =x(2)*x(2);
m(2) =x(2)*x(3);
m(3) =x(2)*x(4);
m(4) =x(2)*x(5);
m(5) =x(3)*x(3);
m(6) =x(3)*x(4);
m(7) =x(3)*x(5);
m(8) =x(4)*x(4);
m(9) =x(4)*x(5);
m(10) =x(5)*x(5);
m(11) =x(2)*m(1);
m(12) =m(1)*x(3);
m(13) =m(1)*x(5);
m(14) =x(3)*m(2);
m(15) =m(2)*x(4);
m(16) =m(2)*x(5);
m(17) =m(2)*x(6);
m(18) =x(5)*m(3);
m(19) =m(3)*x(6);
m(20) =x(5)*m(4);
m(21) =m(4)*x(6);
m(22) =x(3)*m(5);
m(23) =m(5)*x(4);
m(24) =m(5)*x(5);
m(25) =x(4)*m(6);
m(26) =m(6)*x(5);
m(27) =m(6)*x(6);
m(28) =x(5)*m(7);
m(29) =m(7)*x(6);
m(30) =x(4)*m(8);
m(31) =m(8)*x(5);
m(32) =m(8)*x(6);
m(33) =x(5)*m(9);
m(34) =m(9)*x(6);
m(35) =x(6)*x(4)*x(6);
m(36) =x(5)*m(10);
m(37) =m(10)*x(6);
m(38) =x(6)*x(5)*x(6);
m(39) =x(6)*x(6)*x(6);
m(40) =m(11)*x(1);
m(41) =m(22)*x(1);
m(42) =m(30)*x(1);
m(43) =m(36)*x(1);
m(44) =m(39)*x(1);
m(45) =m(11)*x(2);
m(46) =m(12)*x(5);
m(47) =m(14)*x(5);
m(48) =m(16)*x(5);
m(49) =m(22)*x(3);
m(50) =m(22)*x(4);
m(51) =m(22)*x(5);
m(52) =m(23)*x(4);
m(53) =m(24)*x(5);
m(54) =m(25)*x(4);
m(55) =m(28)*x(5);
m(56) =m(30)*x(4);
m(57) =m(30)*x(5);
m(58) =m(31)*x(5);
m(59) =m(31)*x(6);
m(60) =m(33)*x(5);
m(61) =m(33)*x(6);
m(62) =m(34)*x(6);
m(63) =m(36)*x(5);
m(64) =m(39)*x(6);
m(65) =pCoef(2)*m(13)+pCoef(4)*m(20)+pCoef(7)*m(34)+pCoef(8)*m(36);
m(66) =pCoef(3)*m(16)+pCoef(6)*m(28);
m(67) =pCoef(5)*m(24);
m(68) =pCoef(11)*m(13)+pCoef(14)*m(20)+pCoef(22)*m(30)+pCoef(23)*m(36);
m(69) =pCoef(13)*m(16)+pCoef(20)*m(25)+pCoef(21)*m(28);
m(70) =pCoef(18)*m(23)+pCoef(19)*m(24);
m(71) =pCoef(17)*m(22)+pCoef(26)*m(41);
m(72) =pCoef(29)*m(22)+pCoef(36)*m(36)+pCoef(37)*m(37)+pCoef(38)*m(38);
m(73) =pCoef(30)*m(23)+pCoef(34)*m(33)+pCoef(35)*m(34);
m(74) =pCoef(31)*m(25)+pCoef(33)*m(31);
m(75) =pCoef(32)*m(30)+pCoef(43)*m(42);
m(76) =pCoef(46)*m(12)+pCoef(48)*m(14)+pCoef(52)*m(22)+pCoef(56)*m(30)+pCoef(59)*m(32)+pCoef(63)*m(35);
m(77) =pCoef(50)*m(16)+pCoef(53)*m(24)+pCoef(58)*m(31)+pCoef(62)*m(34);
m(78) =pCoef(54)*m(28)+pCoef(61)*m(33);
m(79) =pCoef(65)*m(36)+pCoef(66)*m(43);
m(80) =pCoef(67)*m(16)+pCoef(71)*m(36)+pCoef(72)*m(37)+pCoef(73)*m(38);
m(81) =pCoef(69)*m(33)+pCoef(70)*m(34);
m(82) =pCoef(68)*m(31);
m(83) =pCoef(76);
polyv = zeros(6,1);
polyv(1) =x(3)*m(65)+x(3)*m(66)+x(3)*m(67)+pCoef(1)*m(45)+pCoef(9)*m(40)*x(2);
polyv(2) =x(3)*m(68)+x(3)*m(69)+x(3)*m(70)+x(3)*m(71)+pCoef(10)*m(11)*x(5)+pCoef(12)*m(13)*x(5)+pCoef(15)*m(18)*x(6)+pCoef(16)*m(20)*x(5)+pCoef(24)*m(56)+pCoef(25)*m(63);
polyv(3) =x(4)*m(72)+x(4)*m(73)+x(4)*m(74)+x(4)*m(75)+pCoef(27)*m(16)*x(6)+pCoef(28)*m(49)+pCoef(39)*m(63)+pCoef(40)*m(36)*x(6)+pCoef(41)*m(37)*x(6)+pCoef(42)*m(38)*x(6);
polyv(4) =x(5)*m(76)+x(5)*m(77)+x(5)*m(78)+x(5)*m(79)+pCoef(44)*m(11)*x(3)+pCoef(45)*m(12)*x(3)+pCoef(47)*m(14)*x(3)+pCoef(49)*m(15)*x(6)+pCoef(51)*m(49)+pCoef(55)*m(56)+pCoef(57)*m(30)*x(6)+pCoef(60)*m(32)*x(6)+pCoef(64)*m(35)*x(6);
polyv(5) =x(4)*m(80)+x(4)*m(81)+x(4)*m(82)+pCoef(74)*m(64)+pCoef(75)*m(44)*x(6);
polyv(6) =x(2)*m(83)+pCoef(77)*x(3)+pCoef(78)*x(4)+pCoef(79)*x(5)+pCoef(80)*x(6)+pCoef(81);
jacob = zeros(6,6);
jacob(1,1) = pCoef(9)*m(45);
jacob(1,2) = pCoef(1)*4*m(11) + pCoef(2)*2*m(16) + pCoef(3)*m(24) + pCoef(4)*m(28) + pCoef(9)*4*m(40);
jacob(1,3) =m(65)+2*m(66)+3*m(67);
jacob(1,4) = pCoef(7)*m(29);
jacob(1,5) = pCoef(2)*m(12) + pCoef(3)*m(14) + pCoef(4)*2*m(16) + pCoef(5)*m(22) + pCoef(6)*2*m(24) + pCoef(7)*m(27) + pCoef(8)*3*m(28);
jacob(1,6) = pCoef(7)*m(26);
jacob(2,1) = pCoef(26)*m(49);
jacob(2,2) = pCoef(10)*3*m(13) + pCoef(11)*2*m(16) + pCoef(12)*2*m(20) + pCoef(13)*m(24) + pCoef(14)*m(28) + pCoef(15)*m(34) + pCoef(16)*m(36);
jacob(2,3) =m(68)+2*m(69)+3*m(70)+4*m(71);
jacob(2,4) = pCoef(15)*m(21) + pCoef(18)*m(22) + pCoef(20)*2*m(23) + pCoef(22)*3*m(25) + pCoef(24)*4*m(30);
jacob(2,5) = pCoef(10)*m(11) + pCoef(11)*m(12) + pCoef(12)*2*m(13) + pCoef(13)*m(14) + pCoef(14)*2*m(16) + pCoef(15)*m(19) + pCoef(16)*3*m(20) + pCoef(19)*m(22) + pCoef(21)*2*m(24) + pCoef(23)*3*m(28) + pCoef(25)*4*m(36);
jacob(2,6) = pCoef(15)*m(18);
jacob(3,1) = pCoef(43)*m(56);
jacob(3,2) = pCoef(27)*m(29);
jacob(3,3) = pCoef(27)*m(21) + pCoef(28)*4*m(22) + pCoef(29)*3*m(23) + pCoef(30)*2*m(25) + pCoef(31)*m(30);
jacob(3,4) =m(72)+2*m(73)+3*m(74)+4*m(75);
jacob(3,5) = pCoef(27)*m(17) + pCoef(33)*m(30) + pCoef(34)*2*m(31) + pCoef(35)*m(32) + pCoef(36)*3*m(33) + pCoef(37)*2*m(34) + pCoef(38)*m(35) + pCoef(39)*4*m(36) + pCoef(40)*3*m(37) + pCoef(41)*2*m(38) + pCoef(42)*m(39);
jacob(3,6) = pCoef(27)*m(16) + pCoef(35)*m(31) + pCoef(37)*m(33) + pCoef(38)*2*m(34) + pCoef(40)*m(36) + pCoef(41)*2*m(37) + pCoef(42)*3*m(38);
jacob(4,1) = pCoef(66)*m(63);
jacob(4,2) = pCoef(44)*3*m(12) + pCoef(45)*2*m(14) + pCoef(46)*2*m(16) + pCoef(47)*m(22) + pCoef(48)*m(24) + pCoef(49)*m(27) + pCoef(50)*m(28);
jacob(4,3) = pCoef(44)*m(11) + pCoef(45)*2*m(12) + pCoef(46)*m(13) + pCoef(47)*3*m(14) + pCoef(48)*2*m(16) + pCoef(49)*m(19) + pCoef(50)*m(20) + pCoef(51)*4*m(22) + pCoef(52)*3*m(24) + pCoef(53)*2*m(28) + pCoef(54)*m(36);
jacob(4,4) = pCoef(49)*m(17) + pCoef(55)*4*m(30) + pCoef(56)*3*m(31) + pCoef(57)*3*m(32) + pCoef(58)*2*m(33) + pCoef(59)*2*m(34) + pCoef(60)*2*m(35) + pCoef(61)*m(36) + pCoef(62)*m(37) + pCoef(63)*m(38) + pCoef(64)*m(39);
jacob(4,5) =m(76)+2*m(77)+3*m(78)+4*m(79);
jacob(4,6) = pCoef(49)*m(15) + pCoef(57)*m(30) + pCoef(59)*m(31) + pCoef(60)*2*m(32) + pCoef(62)*m(33) + pCoef(63)*2*m(34) + pCoef(64)*3*m(35);
jacob(5,1) = pCoef(75)*m(64);
jacob(5,2) = pCoef(67)*m(26);
jacob(5,3) = pCoef(67)*m(18);
jacob(5,4) =m(80)+2*m(81)+3*m(82);
jacob(5,5) = pCoef(67)*m(15) + pCoef(68)*m(30) + pCoef(69)*2*m(31) + pCoef(70)*m(32) + pCoef(71)*3*m(33) + pCoef(72)*2*m(34) + pCoef(73)*m(35);
jacob(5,6) = pCoef(70)*m(31) + pCoef(72)*m(33) + pCoef(73)*2*m(34) + pCoef(74)*4*m(39) + pCoef(75)*4*m(44);
jacob(6,2) =m(83);
jacob(6,3) = pCoef(77);
jacob(6,4) = pCoef(78);
jacob(6,5) = pCoef(79);
jacob(6,6) = pCoef(80);
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
xm_2=x(2:end).^3;
xm_1_b=x(2:end).*xm_2-b;
ax1_L=a.*(x(1)-L);
Q=ax1_L.*xm_1_b;
Q(6) = C*x(2:6)-1;
DQ = zeros(6);
for ia=1:5
	DQ(ia, ia+1) = 4*ax1_L(ia)*xm_2(ia);
end
DQ(1:5, 1) = a.*xm_1_b;
DQ(6,2:6)=C;
end

