"""
This file provides kernel estimation function based on
A Majorize-Minimize algorithm with Huber's majorizer.
"""
using MIRT:pogm_restart

"""
Given an image, estimate the kernel using MM algorithm with Huber's majorizer.
Input:
* `Y` Blurred image
* `k` previous kernel estimate
* `X` previous latent image estimate
* `R` original data-fit function, used only for evaluation
* `grad_q` gradient of the majorizer
* `C` function C = R'(z) / z
* `L_diff1` (estimated) Lipschitz constant of lma1
* `L_diff2` (estimated) Lipschitz constant of lma2

Optional:
* `reg` regularization parameter, default 0.1
* `niter` number of outer iterations, default 10
* `ninner` number of inner iterations (POGM), default 5

Output:
* `x` final iterate
* `out` `[fun(x0,0), fun(x1,1), ..., fun(x_niter,niter)]`
"""
function estm_k_mm(Y::AbstractMatrix{<:Number},
                        k::AbstractMatrix{<:Number},
                        X::AbstractMatrix{<:Number},
                        R::Function,
                        grad_q::Function,
                        C::Function,
                        L_diff1::Real,
                        L_diff2::Real;
                        reg::Real = 0.1,
                        niter::Int = 10,
                        ninner::Int = 5)
                  M, N = size(Y)
                  Mk,Nk = size(k)
                  T_im = diff_map_2d(M, N)
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

                  lma1 = getmap_X(diffX1, Mk, Nk) # op. shape is M * N
                  lma2 = getmap_X(diffX2, Mk, Nk) # op. shape is M * N

                  soft = (z, c) -> sign(z) * max(abs(z) - c, 0)
                  g_prox = (x, c) -> soft.(vec(x), c * reg)
                  cost_fun = (x, iter) -> sum(R.(lma1 * vec(x) - vec(diffY1))) + sum(R.(lma2 * vec(x) - vec(diffY2))) + reg * norm(vec(x), 1)
                  kout = vec(copy(k))
                  cout = Array{Any}(undef, niter+1)
                  cout[1] = cost_fun(kout, 0)
                  nor = x -> (x .- minimum(x)) / (maximum(x) - minimum(x))

                  for i = 1:niter
                        kk = copy(kout)
                        L = maximum(C.(kk)) * (L_diff1 + L_diff2)
                        lma1ky = lma1 * kk - vec(diffY1)
                        lma2ky = lma2 * kk - vec(diffY2)
                        ∇f = x -> lma1' * (grad_q.(lma1 * vec(x) - vec(diffY1), lma1ky)) + lma2' * (grad_q.(lma2 * vec(x) - vec(diffY2), lma2ky))
                        Fcost = x -> 0
                        kout, _ = pogm_restart(kk, Fcost, ∇f, L; mom = :pogm, niter = ninner, restart = :gr, g_prox = g_prox)
                        cout[i+1] = cost_fun(kout, i)
                  end

                  kout = max.(kout, 0)
                  kout = kout / sum(kout)
                  return reshape(kout, Mk, Nk), cout
end
