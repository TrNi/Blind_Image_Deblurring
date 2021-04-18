"""
This file provides script of QQ method tested on Levin's dataset.
"""
using LinearAlgebra
include("include_all.jl")

# Setup saving path and final deconv path
path = "/Users/lizongyu/Desktop/Umich/EECS_556/project/code/"
deblur_mpath = "/Users/lizongyu/Desktop/Umich/EECS_556/project/code/overall_20210328/deblur_m/"
M = 256
N = 256

# Setup img index and kernel index
img_idx = "05"
ker_idx = "01"
xtrue = load(path*"/Levin/truth/im"*img_idx*".jld")["gray"]
xtrue = vcat(xtrue, zeros(1, N-1))
xtrue = hcat(xtrue, zeros(M, 1))
xtrue = xtrue'
xtrue = xtrue / maximum(xtrue)
nrmse_x = x -> norm(x - xtrue) / norm(xtrue) * 100
mse_x = x -> norm(x - xtrue)^2 / (M*N)
psnr_x = x -> 10 * log10(maximum(x)^2 / mse_x(x))

ktrue = load(path*"/Levin/kernel/flit"*ker_idx*".jld")["kernel"]
Mk, Nk = size(ktrue)
ktrue = ktrue / sum(ktrue)
Atrue = getmap_K(ktrue, M, N)
Y = reshape(Atrue * vec(xtrue), M, N)

# For image 05
idx_x = 51:120
idx_y = 161:230

# For image 06 small kernel
# idx_x = 81:190
# idx_y = 187:246
# For image 06 large kernel
# idx_x = 71:200
# idx_y = 127:236
# For image 06 kernel 04&06
# idx_x = 21:120
# idx_y = 21:120
# For image 07
# idx_x = 101:250
# idx_y = 81:180
# For image 08
# idx_x = 41:140
# idx_y = 71:150
patch_Y = Y[idx_x, idx_y]
patch_X = xtrue[idx_x, idx_y]
# jim(patch_X)
# plot(jim(patch_X, "xtrue patch"), jim(patch_Y, "blurred patch"),
#       xlims = (1,70), ylims = (1,70), clims = (0,1))
# savefig(path*"levin_patch_0412.pdf")

T_im = diff_map_2d(M, N)
diffX = T_im*vec(xtrue)
diffX1 = reshape(diffX[1:N*(M-1)],M-1,N)
diffX2 = reshape(diffX[N*(M-1)+1:end],M,N-1)
diffY = T_im*vec(Y)
diffY1 = vcat(reshape(diffY[1:N*(M-1)],M-1,N), zeros(1, N))
diffY2 = hcat(reshape(diffY[N*(M-1)+1:end],M,N-1), zeros(M, 1))

# Alternating minimization, first apply on image patches
Z1_init = diffY1[idx_x, idx_y]
Z2_init = diffY2[idx_x, idx_y]
k_init = ones(Mk, Nk)
k_init = k_init / sum(k_init)
reg1 = 0.01
reg2 = 0.01
regk = 0.5
Z1 = copy(Z1_init)
Z2 = copy(Z2_init)
k = copy(k_init)
Z1_cost = []
Z2_cost = []
k_cost = []

for i = 1:10
      Z1, Z1_c = estm_Z_qingqu(diffY1[idx_x, idx_y], k, Z1; reg = reg1, niter = 50)
      Z1_cost = [Z1_cost; Z1_c]
      Z2, Z2_c = estm_Z_qingqu(diffY2[idx_x, idx_y], k, Z2; reg = reg2, niter = 50)
      Z2_cost = [Z2_cost; Z2_c]
      k, k_c = estm_k_qingqu_l1(Y[idx_x, idx_y], k, Z1, Z2; reg = regk, niter = 50)
      k_cost = [k_cost; k_c]
end
jim(k)
k_clean = copy(k)
k_clean[k .< 0.1 * maximum(k)] .= 0
k_clean = k_clean / sum(k_clean)
jim(k_clean)
k_best = find_ker(k_clean,ktrue)
plot(jim(ktrue), jim(k_best), xlims = (1, Mk), ylims = (1, Nk), clims = (0, maximum(ktrue)))
# Let me know if you find a better reg parameter for final deconv!
ydeconv = deconv_outlier(Y,k_best,0.01,deblur_mpath;reg= 0.005)
ydeconv = clamp.(ydeconv, 0, 1)
jim(ydeconv)
@show nrmse_x(ydeconv)

# Alternating minimization on the whole image
diffYdeconv = T_im*vec(ydeconv)
diffYdeconv1 = vcat(reshape(diffYdeconv[1:N*(M-1)],M-1,N), zeros(1, N))
diffYdeconv2 = hcat(reshape(diffYdeconv[N*(M-1)+1:end],M,N-1), zeros(M, 1))

Z1_init = diffYdeconv1
Z2_init = diffYdeconv2
k_init = k_clean
k_init = k_init / sum(k_init)
reg1 = 0.005
reg2 = 0.005
regk = 0.5
Z1 = copy(Z1_init)
Z2 = copy(Z2_init)
k = copy(k_init)
Z1_cost = []
Z2_cost = []
k_cost = []

for i = 1:5
      Z1, Z1_c = estm_Z_qingqu(diffY1, k, Z1; reg = reg1, niter = 20)
      Z1_cost = [Z1_cost; Z1_c]
      Z2, Z2_c = estm_Z_qingqu(diffY2, k, Z2; reg = reg2, niter = 20)
      Z2_cost = [Z2_cost; Z2_c]
      k, k_c = estm_k_qingqu_l1(Y, k, Z1, Z2; reg = regk, niter = 20)
      # k, k_c = estm_k_qingqu(Y, k, Z1, Z2; niter = 10)
      k_cost = [k_cost; k_c]
end
jim(k)
k_clean = copy(k)
k_clean[k .< 0.1 * maximum(k)] .= 0
k_clean = k_clean / sum(k_clean)
jim(k_clean)
k_best = find_ker(k_clean,ktrue)
plot(jim(ktrue), jim(k_best), xlims = (1, Mk), ylims = (1, Nk), clims = (0, maximum(ktrue)))
ydeconv = deconv_outlier(Y,k_best,0.01,deblur_mpath;reg= 0.005)
ydeconv = clamp.(ydeconv, 0, 1)
nrmse_ydeconv = nrmse_x(ydeconv)
nrmse_blurred = nrmse_x(Y)
psnr_ydeconv = psnr_x(ydeconv)
psnr_blurred = psnr_x(Y)
plot(jim(xtrue, "xtrue"), jim(Y, "blurred image \n NRMSE = "*string(round(nrmse_blurred; digits = 1))*"%"),
      jim(ydeconv, "deblurred image \n NRMSE = "*string(round(nrmse_ydeconv; digits = 2))*"%"),
      xlims = (1, M), ylims = (1, N), clims = (0,1))
savefig(path*"levin_deblur_img"*img_idx*"ker"*ker_idx*".pdf")
save(path*"levin_deblur_img"*img_idx*"ker"*ker_idx*".jld",
      "xtrue", xtrue,
      "ktrue", ktrue,
      "blurred", Y,
      "ydeconv", ydeconv,
      "kout", k_best,
      "reg1", reg1,
      "reg2", reg2,
      "regk", regk,
      "nrmse_blurred", nrmse_blurred,
      "psnr_blurred", psnr_blurred,
      "nrmse_ydeconv", nrmse_ydeconv,
      "psnr_ydeconv", psnr_ydeconv)
