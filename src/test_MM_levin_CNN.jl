include("include_all.jl")

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
path_to_SelfDeblur = joinpath(rootpath, "Code", "Code_in_Julia", "Overall", "SelfDeblur")
path_to_results = joinpath(rootpath, "Code", "Code_in_Julia", "results")
path_to_resultscnn = joinpath(rootpath, "Code", "Code_in_Julia", "Wei", "results_cnnmm")

M = 256
N = 256

a = 43
b = 3
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

psnr_re = zeros(4,8)
nrmse_re = zeros(4,8)

for img_index = 1:4
    # read xtrue
    xtrue = load(joinpath(path_to_Levin, "ground", "im0"*string(img_index+4)*".jld"))["gray"]
    xtrue = vcat(xtrue, zeros(1, N-1))
    xtrue = hcat(xtrue, zeros(M, 1))
    xtrue = xtrue'
    xtrue = xtrue / maximum(xtrue)
    nrmse_x = x -> norm(vec(x) - vec(xtrue)) / norm(vec(xtrue))
    psnr_x = x -> assess_psnr(x, xtrue)

    for k_index = 1:8
        # read kernel
        ktrue = load(joinpath(path_to_Levin, "kernel", "flit0"*string(k_index)*".jld"))["kernel"]
        ktrue = ktrue / sum(ktrue)
        k_size = size(ktrue, 1)
        Atrue = getmap_K(ktrue, M, N)
        Y = reshape(Atrue * vec(xtrue), M, N)

        # read deblur
        deblurred = load(joinpath(path_to_SelfDeblur, "imgs", "im"*string(img_index)*"_kernel"*string(k_index)*"_img_x.png"))
        deblurred = Gray.(deblurred)
        deblurred = convert(Array{Float64}, deblurred) # added
        deblurred = vcat(deblurred, zeros(5, M-1))
        deblurred = vcat(zeros(5, M-1), deblurred)
        deblurred = hcat(deblurred, zeros(M+9, 5))
        deblurred = hcat(zeros(M+9, 5), deblurred)
        deblurred = deblurred'

        # Calculate best shift for CNN result
        nrmse_min = 1
        i_select = 0
        j_select = 0
        deblurred_shift = deblurred
        for i = 1:10
            for j = 1:10
                deblurred_cut = deblurred[i:(i+M-1), j:(j+N-1)]
                deblurred_cut = deblurred_cut / maximum(deblurred_cut)
                nrmse_cut = nrmse_x(deblurred_cut)
                if nrmse_cut<nrmse_min
                    nrmse_min = nrmse_cut
                    i_select = i
                    j_select = j
                    deblurred_shift = deblurred_cut
                end
            end
        end
        deblurred = deblurred_shift

        # mm
        Xout_irls = copy(Y)
        kout_init = zeros(k_size, k_size) + 0.01 * rand(k_size, k_size)
        kout_init = kout_init / sum(kout_init)
        t_max = 2

        L_diff1 = opnorm(Matrix(getmap_psX(diff(deblurred, dims = 1), (k_size, k_size))), 2)
        L_diff2 = opnorm(Matrix(getmap_psX(diff(deblurred, dims = 2), (k_size, k_size))), 2)

        kout_mm, cout_ker_mm = estm_k_mm(Y, kout_init, deblurred, R, grad_q, C, L_diff1, L_diff2;
                                 reg = 1, niter = 10, ninner = 5)

        Xout_mm, cout_im_mm = estm_lt_img_mm(Y, kout_mm, deblurred, R, grad_q, C;
                                 reg = 3e-4, niter = 10, ninner = 5)

        for i = 1:t_max
            L_diff1 = opnorm(Matrix(getmap_psX(diff(Xout_mm, dims = 1), (k_size, k_size))), 2)
            L_diff2 = opnorm(Matrix(getmap_psX(diff(Xout_mm, dims = 2), (k_size, k_size))), 2)
            kout_mm, cout_ker_mm = estm_k_mm(Y, kout_mm, Xout_mm, R, grad_q, C, L_diff1, L_diff2;
                                     reg = 1, niter = 10, ninner = 5)
            Xout_mm, cout_im_mm = estm_lt_img_mm(Y, kout_mm, Xout_mm, R, grad_q, C;
                                     reg = 3e-4, niter = 10, ninner = 5)
        end

        ydeconv = deconv_outlier(Y,kout_mm,0.01,deblur_mpath;reg= 0.001)
        ydeconv = min.(max.(ydeconv, 0),1)

        psnr_re[img_index,k_index,1] = psnr_x(Y)
        psnr_re[img_index,k_index,2] = psnr_x(ydeconv)
        nrmse_re[img_index,k_index,1] = nrmse_x(Y)
        nrmse_re[img_index,k_index,2] = nrmse_x(ydeconv)

        print(nrmse_x(ydeconv))
        sleep(10)

        save(joinpath(path_to_resultscnn,"img"*string(img_index)*"_k"*string(k_index)*"_est_k.jld"),"kernel",kout_mm)
        save(joinpath(path_to_resultscnn,"img"*string(img_index)*"_k"*string(k_index)*"_est_img.jld"),"gray",ydeconv)

    end
end

save(joinpath(path_to_resultscnn,"cnn_mm_re.jld"),"psnr",psnr_re,"nrmse",nrmse_re)

data = load(joinpath(path_to_resultscnn,"cnn_mm_re.jld"))
