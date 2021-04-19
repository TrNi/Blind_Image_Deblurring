using LinearAlgebra
include("include_all.jl")
include("pyramid.jl")
include("im_estm_mm_diff.jl")

# For others to run directly on google drive
pathprefix=""
if Sys.iswindows()
    pathprefix = "G:\\"
elseif Sys.isapple()
    pathprefix = joinpath("/Volumes","GoogleDrive")
else
    nothing # insert path prefix for your operating system.
end

rootpath= joinpath(pathprefix,"Shared drives","EECS 556 - I.P. Project")
deblur_mpath = joinpath(rootpath,"Code","Code_in_Julia","Overall","deblur_m")
path_to_Levin = joinpath(rootpath, "Data", "img", "Levin")
path_to_resultspmm = joinpath(rootpath, "Code", "Code_in_Julia", "Wei", "results_pmm")
# path_to_resultscnn = joinpath(rootpath, "Code", "Code_in_Julia", "Wei", "results_cnnmm")

M = 256
N = 256

a = 43
b = 2
R(z) = isinf(exp(b*z^2)) ? 0 : z^2/2 - log(a+exp(b*z^2))/(2*b) + log(a+1)/(2*b)
grad_R(z) = isinf(exp(b*z^2)) ? 0 : z - 1/(2*b)*1/(a + exp(b*z^2)) * exp(b*z^2)*2*z*b

function C(z)
    if z == 0
        return a/(a+1)
    elseif isnan(grad_R(z)/z)
        return 0
    else
        return max(grad_R(z)/z, 0)
    end
end
q(x, z) = C(z)/2 * (x - z)^2 + grad_R(z) * (x - z) + R(z)
grad_q(x, z) = C(z) * (x - z) + grad_R(z)


xtrue = load(joinpath(path_to_Levin, "ground", "im05.jld"))["gray"]
xtrue = vcat(xtrue, zeros(1, N-1))
xtrue = hcat(xtrue, zeros(M, 1))
xtrue = xtrue'
xtrue = xtrue / maximum(xtrue)

nrmse_x = x -> norm(x - xtrue) / norm(xtrue)
psnr_x = x -> (assess_psnr(x,xtrue))

ktrue = load(joinpath(path_to_Levin, "kernel", "flit01.jld"))["kernel"]
Mk, Nk = size(ktrue)
ktrue = ktrue / sum(ktrue)
Atrue = getmap_K(ktrue, M, N)
Y = reshape(Atrue * vec(xtrue), M, N)
plot(jim(xtrue), jim(Y))

# idx_x = 53:116
# idx_y = 163:226
idx_x = (50:119); idx_y = (160:229) # for img5
# idx_x = (101:250); idx_y = (81:180) # for img7
# idx_x = (151:250); idx_y = (1:100) # for img7 final used
# idx_x = (141:240); idx_y = (141:240) # for img7
# idx_x = (50:119) .-20 ; idx_y = (160:229) .+10
patch_Y = Y[idx_x, idx_y]
patch_X = xtrue[idx_x, idx_y]
plot(jim(patch_X, "xtrue patch"), jim(patch_Y, "blurred patch"), clims = (0,1))

k_init = ones(Mk, Nk)
k_init = k_init / sum(k_init)
X_init = copy(patch_Y)

reg1 = 0.3
reg2 = 0.01
regk = 0.8
kout_mm = copy(k_init)
Xout_mm = copy(X_init)
Xout_cost = []
k_cost = []

for i = 1:3
    Xout_mm, cout_im_mm = estm_lt_img_mm_diff(patch_Y, kout_mm, Xout_mm, R, grad_q, C;
                             reg1 = reg1, reg2 = reg2, niter = 5, ninner = 5)
    Xout_cost = [Xout_cost; cout_im_mm]
    kout_mm, cout_ker_mm = estm_k_mm(patch_Y, kout_mm, Xout_mm, R, grad_q, C;
                             reg = regk, niter = 5, ninner = 5)
    k_cost = [k_cost; cout_ker_mm]
end

kout_mm = kout_mm / sum(kout_mm)
k_clean = copy(kout_mm)
k_clean[kout_mm .< 0.1 * maximum(kout_mm)] .= 0
k_clean = k_clean / sum(k_clean)
k_best = find_ker(k_clean,ktrue)
k_best = k_best/sum(k_best)
plot(jim(ktrue), jim(kout_mm), jim(k_best), clims = (0, maximum(ktrue)))

deblurred = deconv_outlier(Y,k_best,0.01,deblur_mpath;reg= 0.01)
deblurred = max.(min.(deblurred, 1),0)
jim(deblurred)
@show nrmse_x(Y), nrmse_x(deblurred)


Xout_init = copy(deblurred)
kout_init = copy(k_best)

kout_mm = copy(kout_init)
Xout_mm = copy(Xout_init)
t_max = 1

k_cost = []
Xout_cost = []

for i = 1:4
    kout_mm, cout_ker_mm = estm_k_mm(Y, kout_mm, Xout_mm, R, grad_q, C;
                            reg = 1, niter = 10, ninner = 5)
    k_cost = [k_cost; cout_ker_mm]
    Xout_mm, cout_im_mm = estm_lt_img_mm(Y, kout_mm, Xout_mm, R, grad_q, C;
                            reg = 3e-4, niter = 10, ninner = 5)
    Xout_cost = [Xout_cost; cout_im_mm]
end

jim(Xout_mm)
plot(k_cost)
plot(Xout_cost)
plot(jim(ktrue), jim(kout_mm), clims = (0, 0.1))

k_clean = copy(kout_mm)
k_clean[kout_mm .< 0.1 * maximum(kout_mm)] .= 0
k_clean = k_clean / sum(k_clean)
k_best = find_ker(k_clean,ktrue)
k_best = k_best/sum(k_best)
jim(k_best)

ydeconv = deconv_outlier(Y,k_best,0.01,deblur_mpath;reg= 0.001)
ydeconv = min.(max.(ydeconv, 0),1)
jim(ydeconv)

@show nrmse_x(Y), psnr_x(Y), nrmse_x(ydeconv), psnr_x(ydeconv)

# save(joinpath(path_to_resultspmm,"img07_k8_est_k.jld"),"kernel",k_best)
# save(joinpath(path_to_resultspmm,"img07_k8_est_img.jld"),"gray",ydeconv)
