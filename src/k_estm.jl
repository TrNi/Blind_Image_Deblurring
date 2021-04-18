"""
This file provides kernel estimation function based on
the IRLS kernel Estimation algorithm with l1-norm prior,
proposed by Dong et al.
"""
# include("diff2d_forw_and_adj.jl")
# include("linearmap_conv.jl")
# include("ncg_inv_zl.jl")
"""
Given an image, estimate the kernel using
IRLS kernel Estimation algorithm with l1-norm prior,
proposed by Dong et al.

Input:
* `Y` Blurred image
* `k` previous kernel estimate
* `X` previous latent image estimate
* `C` function C = R'(z) / z

Optional:
* `reg` regularization parameter, optional
* `niter` number of iterations to run, default 10
* `thresh` threshold to stop inversion at, default 1e-5

Output:
* `x` final iterate
* `out` `[fun(x0,0), fun(x1,1), ..., fun(x_niter,niter)]`
"""
function estm_k(Y::AbstractMatrix{<:Number},
				k::AbstractMatrix{<:Number},
                X::AbstractMatrix{<:Number},
                C::Function;
                reg::Real = 0.1,
                niter::Int = 10,
				thresh::Real = 1e-5)
              M, N = size(Y)
              Mk,Nk = size(k)
              T_im = diff_map_2d(M, N) # 2D finite difference linear map.
              # T*vec(X) is computed only once and its outputs are partitioned.

              diffX = T_im*vec(X)
              diffY = T_im*vec(Y)
              diffX1 = reshape(diffX[1:N*(M-1)],M-1,N)
              diffX2 = reshape(diffX[N*(M-1)+1:end],M,N-1)
			  diffY1 = reshape(diffY[1:N*(M-1)],M-1,N)
              diffY2 = reshape(diffY[N*(M-1)+1:end],M,N-1)

			  # Padding is needed for constructing conv map from images, we need even images.
			  diffX1 = vcat(diffX1, zeros(1, N))
			  diffX2 = hcat(diffX2, zeros(M, 1))
			  diffY1 = vcat(diffY1, zeros(1, N))
			  diffY2 = hcat(diffY2, zeros(M, 1))

			  lma1 = getmap_X(diffX1,Mk,Nk) # op shape is M*N after above padding.
              lma2 = getmap_X(diffX2,Mk,Nk) # op shape is M*N.
		   	  my_exp = x -> (abs(x) > thresh) ? x^(-1) : thresh^(-1)

              W_R1 = C.(lma1*vec(k) - vec(diffY1))
              W_R2 = C.(lma2*vec(k) - vec(diffY2))
              W_G = my_exp.(abs.(vec(k)))

              cost_fun = (vark, iter) -> 1/2 * norm(sqrt.(W_R1) .* (lma1*vec(vark) - vec(diffY1)))^2 +
                                            1/2 * norm(sqrt.(W_R2) .* (lma2*vec(vark) - vec(diffY2)))^2 +
                                            reg/2 * norm(sqrt.(W_G) .* (vec(vark)))^2

              B = [lma1, lma2, I(Mk*Nk)]
              gf = [x -> vec(x) - vec(diffY1), x -> vec(x) - vec(diffY2), x -> reg * vec(x)]
              W = [W_R1, W_R2, W_G]
              regg = [1 , 1, reg]
              kout, cout = ncg_inv_zl(B, gf, W, regg, vec(k); niter = niter, fun = cost_fun)
			  kout = max.(kout, 0)
			  kout = kout / sum(kout)
        return reshape(kout, Mk, Nk), cout
end
