"""
This file provides image estimation function based on
the IRLS Image Estimation algorithm with Hyper-Laplacian priors,
proposed by Dong et al.
"""
# include("diff2d_forw_and_adj.jl")
# include("linearmap_conv.jl")
# include("ncg_inv_zl.jl")

"""
Given a kernel, estimate the latent image using
IRLS Image Estimation algorithm proposed by Dong et al.

Input:
* `Y` Blurred image
* `k` initial estimate of the kernel
* `X` estimate in the previous iteration
* `C` function C = R'(z) / z

Optional:
* `reg` regularization parameter, default 0.008
* `niter` number of iterations to run, default 10
* `thresh` threshold to stop inversion at, default 1e-5

Output:
* `x` final iterate
* `out` `[fun(x0,0), fun(x1,1), ..., fun(x_niter,niter)]`
"""
function estm_lt_img(Y::AbstractMatrix{<:Number},
                        k::AbstractMatrix{<:Number},
                        X::AbstractMatrix{<:Number},
                        C::Function;
                        reg::Real = 0.008,
                        niter::Int = 10,
                        thresh::Real = 1e-5)
                  M, N = size(Y)
                  A = getmap_K(k, M, N)
                  T = diff_map_2d(M, N)
                  W_R = C.(A*vec(X) - vec(Y))
                  my_exp = x -> (abs(x) > thresh) ? x^(-1.2) : thresh^(-1.2)
                  W_G = my_exp.(abs.(T * vec(X)))
                  cost_fun = (x, iter) -> 1/2 * norm(sqrt.(W_R) .* (A*vec(x) - vec(Y)))^2 +
                                         reg/2 * norm(sqrt.(W_G) .* (T * vec(x)))^2
                  B = [A, T]
                  gf = [x -> vec(x) - vec(Y), x -> reg * vec(x)]
                  W = [W_R, W_G]
                  regg = [1 , reg]
                  xout, cout = ncg_inv_zl(B, gf, W, regg, vec(X); niter = niter, fun = cost_fun)
            return reshape(xout, M, N), cout
end
