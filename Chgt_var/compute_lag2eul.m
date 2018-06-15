clearvars;

%% bools
compute_grad_rho_xi = false;
compute_grad_p0 = false;
compute_grad_v0 = true;
output_montjoie = true;

%% pulsation and damping
w = 2*pi*0.78;
sigma = 0.1;

%% loading datas
dir_name = "totalTest2/";
num_comp = 2;
files = ["rho0","M_0","M_1","c0","grad_rho0_0","grad_rho0_1","grad_p0_0","grad_p0_1"];
for i = 0:num_comp-1
    files = [files, strcat("U",num2str(i))];
end
for i = 0:2*num_comp-1
    files = [files, strcat("dU",num2str(i))];
end
for f=files
    [x,y,z,coor,tmp]=loadND(strcat(dir_name,f,".dat"));
    eval(strcat(f,"=tmp;"));
end
clear dir_name num_comp files f i tmp

%% computing eulerian perturbation for pressure
if compute_grad_p0
    [grad_p0_0, grad_p0_1] = gradient(p0);
end
p_E = -rho0.*c0.*c0.*(dU0+dU2)+U0.*grad_p0_0+U1.*grad_p0_1;

%% computing eulerian perturbation for density
if compute_grad_rho_xi
    rho_xi_0 = rho0.*U0;
    rho_xi_1 = rho0.*U1;
    [grad_rx0_0, ~] = gradient(rho_xi_0);
    [~, grad_rx1_1] = gradient(rho_xi_1);
    rho_E = -grad_rx0_0-grad_rx1_1;
    clear rho_xi_0 rho_xi_1 grad_rx0_0 grad_rx1_1
else
    rho_E = rho0.*(dU0+dU2)+U0.*grad_rho0_0+U1.*grad_rho0_1;
end

%% computing eulerian perturbation for velocity
if compute_grad_v0
    [grad_M_0, grad_M_1] = gradient(M_0);
    [grad_M_2, grad_M_3] = gradient(M_1);
end
M_grad_xi_0 = M_0.*dU0+M_1.*dU1;
M_grad_xi_1 = M_0.*dU2+M_1.*dU3;
DDt_xi_0 = (-w*1i+sigma).*U0+M_grad_xi_0;
DDt_xi_1 = (-w*1i+sigma).*U1+M_grad_xi_1;

v_E_0 = DDt_xi_0-U0.*grad_M_0-U1.*grad_M_1;
v_E_1 = DDt_xi_1-U0.*grad_M_2-U1.*grad_M_3;

clear M_grad_xi_0 M_grad_xi_1 DDt_xi_0 DDt_xi_1

%% output
if output_montjoie
    v_E_0 = rho0.*v_E_0;
    v_E_1 = rho0.*v_E_1;
end

