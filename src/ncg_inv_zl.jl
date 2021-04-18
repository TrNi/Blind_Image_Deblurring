"""
(x, out) = ncg_inv_zl(B, gf, W, x0; [options])

Nonlinear preconditioned conjugate gradient algorithm
to minimize a general "inverse problem" cost function `sum_{j=1}^J f_j(B_j x)`
where each `f_j` is a weighted quadratic function.

In
* `B` array of `J` blocks `B_1, ..., B_J`
* `gf` array of `J` functions for computing gradients of `f_1, ..., f_J`
* `W` array of `J` blocks `W_1, ..., W_J`
* `reg` array of `J` regularization parameters
* `x0` initial guess

Option
* `niter` # number of outer iterations; default `50`
* `P` preconditioner; default `I`
* `betahow` "beta" method for the search direction: default `:dai_yuan`
* `fun` User-defined function to be evaluated with two arguments `(x,iter).
It is evaluated at `(x0,0)` and then after each iteration.

Out
* `x` final iterate
* `out` `[fun(x0,0), fun(x1,1), ..., fun(x_niter,niter)]`
"""

function ncg_inv_zl(
                B::AbstractVector{<:Any},
                gf::AbstractVector{<:Function},
                W::AbstractVector{<:Any},
                reg::AbstractVector{<:Number},
                x0::AbstractVector{<:Number};
                niter::Int = 50,
                P = I,
                betahow::Symbol = :daiyuan,
                fun::Function = (x,iter) -> undef)

        out = Array{Any}(undef, niter + 1)
        out[1] = fun(x0, 0)

        J = length(B)

        x = x0
        dir = []
        grad_old = []
        grad_new = []

        Bx = [(B[j] * x) for j = 1:J]
        grad = (Bx) -> sum([(B[j]' * (W[j] .* gf[j](Bx[j]))) for j = 1:J])

        for iter = 1:niter
                grad_new = grad(Bx)
                npgrad = -(P * grad_new)
                if iter == 1
                        dir = npgrad
                else
                        if betahow === :daiyuan
                                betaval = grad_new' * (P * grad_new) / ((grad_new - grad_old)' * dir)
                        else
                                throw("unknown beta choice")
                        end
                        dir = npgrad + betaval * dir # search direction
                end
                grad_old = grad_new

                Bd = [(B[j] * dir) for j = 1:J]

                alf_sol = - sum([(dir' * (B[j]' * (W[j] .* gf[j](Bx[j])))) for j = 1:J]) /
                                sum([(reg[j] * (Bd[j]' * (W[j] .* Bd[j]))) for j = 1:J])

                if isnan(alf_sol)
                        print("break due to nan!")
                        break
                end

                x += alf_sol * dir
                Bx += alf_sol * Bd
                out[iter + 1] = fun(x, iter)
        end

        return x, out
end
