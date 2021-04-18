using MIRT:pogm_restart
"""
Given an image, estimate the kernel using POGM algorithm.
Input:
* `Y` Blurred image
* `k` previous kernel estimate
* `Z1` previous latent gradient estimate along X direction
* `Z2` previous latent gradient estimate along Y direction
* `reg` regularization parameter, optional
* `regfunc` function of regularization, optional
* `niter` number of outer iterations (POGM)

Output:
* `kout` final iterate
* `cout` `[fun(kout0,0), fun(kout1,1), ..., fun(kout_niter,niter)]`
"""
function estm_k_qingqu_l1(Y::AbstractMatrix{<:Number},
                        k::AbstractMatrix{<:Number},
                        Z1::AbstractMatrix{<:Number},
                        Z2::AbstractMatrix{<:Number};
                        reg::Real = 0.1,
						regfunc::Symbol = :sum,
                        niter::Int = 50)


                  M, N = size(Y)
                  Mk,Nk = size(k)

                  T_im = diff_map_2d(M, N)

                  diffY = T_im*vec(Y)

                  diffY1 = reshape(diffY[1:N*(M-1)],M-1,N)
                  diffY2 = reshape(diffY[N*(M-1)+1:end],M,N-1)

                  # Padding is needed for constructing conv map from images, we need even images.
                  diffY1 = vcat(diffY1, zeros(1, N))
                  diffY2 = hcat(diffY2, zeros(M, 1))

                  lma1 = getmap_X(Z1, Mk, Nk)
                  lma2 = getmap_X(Z2, Mk, Nk)


                  Fcost = x -> 1/2 * norm(vec(diffY1) - lma1 * vec(x), 2)^2 +
                                          1/2 * norm(vec(diffY2) - lma2 * vec(x), 2)^2 +
                                          reg * norm(vec(x), 1)

                  L = opnorm(Matrix(lma1), 2)^2 + opnorm(Matrix(lma2), 2)^2
                  ∇f = x -> lma1' * (lma1 * vec(x) - vec(diffY1)) + lma2' * (lma2 * vec(x) - vec(diffY2))
                  soft = (z, c) -> sign(z) * max(abs(z) - c, 0)
				  g_prox = (z, c) -> soft.(z, reg * c)
				  cost_fun = (iter, xk, zk, is_restart)-> Fcost(xk)

				  kout, cout = pogm_restart(vec(k), x -> [], ∇f, L;
								mom = :pogm, niter = niter, restart = :gr, g_prox = g_prox, fun = cost_fun)
				  kout = max.(kout, 0)
				  if regfunc === :sum
					  kout = kout / sum(kout)
				  elseif regfunc === :norm
					  kout = kout / norm(kout, 2)
				  else
					  throw("unknown regfunc")
				  end
            return reshape(kout, Mk, Nk), cout
end
