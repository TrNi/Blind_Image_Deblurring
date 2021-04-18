"""
This file provides image estimation function based on
a Majorize-Minimize algorithm with Huber's majorizer.
"""
using MIRT:Aodwt, pogm_restart

"""
Given a kernel, estimate the latent image using
a Majorize-Minimize algorithm with Huber's majorizer
and ODWT regularization.

NOTE: The image must be square and its size must be exponentials of 2.

Input:
* `Y` Blurred image
* `k` initial estimate of the kernel
* `X` estimate in the previous iteration
* `R` original data-fit function, used only for evaluation
* `grad_q` gradient of the majorizer
* `C` function C = R'(z) / z

Optional:
* `reg` regularization parameter, default 0.1
* `niter` number of outer iterations, default 10
* `ninner` number of inner iterations (POGM), default 5

Output:
* `x` final iterate
* `out` `[fun(x0,0), fun(x1,1), ..., fun(x_niter,niter)]`
"""
function estm_lt_img_mm(Y::AbstractMatrix{<:Number},
                        k::AbstractMatrix{<:Number},
                        X::AbstractMatrix{<:Number},
                        R::Function,
                        grad_q::Function,
                        C::Function;
                        reg::Real = 0.008,
                        niter::Int = 10,
                        ninner::Int = 5)

                  # Size of the image
                  M, N = size(Y)

                  A = getmap_K(k, M, N)

                  T, scales, mfun = Aodwt((M, N)) # T is real
                  soft = (z, c) -> sign(z) * max(abs(z) - c, 0)
                  g_prox = (x, c) -> vec(T' * reshape(soft.(vec(T * reshape(x,M,N)), c * reg * (scales[:] .!= 0)), M, N))
                  cost_fun = (x, iter) -> sum(R.(A * vec(x) - vec(Y))) + reg * norm(vec(T * reshape(x,M,N)), 1)
                  xout = vec(copy(X))
                  cout = Array{Any}(undef, niter+1)
                  cout[1] = cost_fun(X, 0)
                  nor = x -> min.(max.(x, 0), 1)

                  for i = 1:niter
                        xk = copy(xout)
                        L = maximum(C.(xk))
                        Axky = A * xk - vec(Y)
                        ∇f = x -> A' * (grad_q.(A * vec(x) - vec(Y), Axky))
                        Fcost = x -> 0
                        xout, _ = pogm_restart(xk, Fcost, ∇f, L; mom = :pogm, niter = ninner, restart = :gr, g_prox = g_prox)
                        xout = nor(xout)
                        cout[i+1] = cost_fun(xout, i)
                  end

            return reshape(xout, M, N), cout
end
