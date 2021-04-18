# include("diff2d_forw_and_adj.jl")
# # include("fft_conv_nt.jl")
# include("linearmap_conv.jl")
# include("ncg_inv_zl.jl")

#sqrtc = x -> sqrt(Complex(x)) # sqrt of complex cast input.
# zeropad = (X,m1,n1,m2,n2) ->padarray(X,Fill(0,(m1,n1),(m2,n2)))
# Fill(x,(a,b),(c,d)) adds 'a' rows, 'b' columns before x;
# and 'c' rows, 'd' columns after x.
"""
Given an image, estimate the kernel.
Input:
* `Y` Blurred image
* `k` previous kernel estimate
* `X` previous latent image estimate
* `C` function C = R'(z) / z
* `reg` regularization parameter, optional

Output:
* `x` final iterate
* `out` `[fun(x0,0), fun(x1,1), ..., fun(x_niter,niter)]`
"""
function estm_k_simple(Y::AbstractMatrix{<:Number},
                        k::AbstractMatrix{<:Number},
                        X::AbstractMatrix{<:Number},
                        C::Function;
                        reg::Real = 0.1,
                        niter::Int = 10,
						thresh::Real = 1e-5)
                  # Y = Float16.(Y);X = Float16.(X);k = Float16.(k); #remove this only after casting all images.
                  M, N = size(Y)
                  Mk,Nk = size(k)
				  A = getmap_X(reshape(X, M, N),Mk, Nk)
				  my_exp = x -> (abs(x) > thresh) ? x^(-1) : thresh^(-1)
				  W_R = C.(A*vec(k) - vec(Y))
                  W_G = my_exp.(abs.(vec(k)))
				  cost_fun = (vark, iter) -> 1/2 * norm(sqrt.(W_R) .* (A*vec(vark) - vec(Y)))^2 +
                                            reg/2 * norm(sqrt.(W_G) .* (vec(vark)))^2
                  B = [A, I(Mk*Nk)]
                  gf = [x -> vec(x) - vec(Y), x -> reg * vec(x)]
                  W = [W_R, W_G]
                  regg = [1, reg]
				  # Caution: Check what you should initialize with - image or kernel!
                  kout, cout = ncg_inv_zl(B, gf, W, regg, vec(k); niter = niter, fun = cost_fun)
				  kout = max.(kout, 0)
				  kout = kout / sum(kout)
            return reshape(kout, Mk, Nk), cout
end


# Kout, cout = estm_k(Y[20:40,20:40], kernel_init[2:5,2:5], X_init[20:40,20:40], C; reg = 0.008,niter=1)
# Kout, cout = estm_k(Y, kernel_init, X_init, C; reg = 0.008,niter=1)
# Kout, cout = estm_k(Y[20:220,20:220], kernel_init[2:5,2:5], X_init[20:220,20:220], C; reg = 0.008,niter=10)
