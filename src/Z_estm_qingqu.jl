using MIRT:pogm_restart
"""
Given a kernel, estimate the latent image using POGM algorithm.
Input:
* `∇Y` Gradient of blurred image
* `k` previous kernel estimate
* `Z` previous latent gradient estimate
* `reg` regularization parameter, optional
* `niter` number of outer iterations (POGM)

Output:
* `Zout` final iterate
* `cout` `[fun(Zout0,0), fun(Zout1,1), ..., fun(Zout_niter,niter)]`
"""
function estm_Z_qingqu(∇Y::AbstractMatrix{<:Number},
					k::AbstractMatrix{<:Number},
                	Z::AbstractMatrix{<:Number};
                	reg::Real = 0.1,
                	niter::Int = 50)

			M, N = size(∇Y)
			A = getmap_K(k, M, N)
			soft = (z, c) -> sign(z) * max(abs(z) - c, 0)
			g_prox = (z, c) -> soft.(z, reg * c)
			Fcost = x -> 1/2 * norm(vec(∇Y) - A * vec(x), 2)^2 + reg * norm(vec(x), 1)
			∇f = x -> A' * (A * vec(x) - vec(∇Y))
			L = sum(vec(k))^2
			cost_fun = (iter, xk, zk, is_restart)-> Fcost(xk)

			Zout, cout = pogm_restart(vec(Z), x -> [], ∇f, L;
				mom = :pogm, niter = niter, restart = :gr, g_prox = g_prox, fun = cost_fun)

			return reshape(Zout, M, N), cout
end
