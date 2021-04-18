include("im_estm.jl")
include("k_estm.jl")
"""
Given an image, alternatively and iteratively estimate the latent image and kernel
using the IRLS algorithm with Hyper-Laplacian and l1-norm priors, proposed by Dong et al.

Input:
* `Y` Blurred image
* `k` initial kernel estimate
* `C` function C = R'(z) / z

Optional:
* `reg_im` regularization parameter for latent image, default 0.008
* `reg_k` regularization parameter for kernel, default 0.1
* `niter_img` number of iterations for latent image estimation, default 50
* `niter_k` number of iterations for kernel estimation, default 50
* `niter_all` number of iterations for alternating minimization, default 1

Output:
* `X` final estimated latent image
* `k` final estimated kernel
* `lt_cost` cost of latent image estimation as a vector of length niter_alt * niter_img
* `k_cost` cost of kernel estimation as a vector of length niter_alt * niter_k
* `kernels` a list of kernels obtained in each outer loop, with dimension kernel_size x kernel_size x niter_alt
"""

function IRLS(Y::AbstractMatrix{<:Number},
                  k::AbstractMatrix{<:Number},
                  C::Function;
                  X=nothing, niter_im=50, niter_k=50, niter_all=1,
                  reg_k=0.1, reg_im=0.008)

    lt_cost = []
    k_cost = []
    if X == nothing
        X = Y
    end

    for i = 1:niter_all
        X, cout = estm_lt_img(Y, k, X, C; niter=niter_im, reg=reg_im)
        lt_cost = [lt_cost;cout]
        k, cout = estm_k(Y, k, X, C; niter=niter_k, reg=reg_k)
        k[abs.(k) .< 0.1 * maximum(k)] .= 0
        k = max.(k, 0)
        k = k / sum(k)
        k_cost = [k_cost;cout]
        display(plot(jim(X), jim(k)))
    end
    return lt_cost, k_cost, X, k
end
