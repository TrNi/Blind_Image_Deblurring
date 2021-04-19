using LinearMapsAA
function diff2d_forw(x::AbstractMatrix)
    d=[diff(x,dims=1)[:];diff(x,dims=2)[:]]
    return d
end

function diff2d_adj(d::AbstractVector{<:Number},M::Int, N::Int ; out2d::Bool=false)
    #adj_diff = y -> [-y[1]; -diff(y) ; y[end] ] # adjoint(C) * 1-D vector y.
    #To understand: process d1 and d2 individually, put the outputs in matrix form.
    d1=reshape(d[1:N*(M-1)],M-1,N)
    d2=reshape(d[N*(M-1)+1:end],M,N-1) #N-1 x M
    #@show size(d1),size(d2)
    z1=vcat(transpose(-d1[1,:]),-diff(d1,dims=1),transpose(d1[end,:]))[:]
    z2=[-d2[:,1][:]; -diff(d2,dims=2)[:];d2[:,end][:]]
    #@show size(z1),size(z2)
    z=z1+z2
    if out2d == true
        z=reshape(z,M,N)
    end
    return z
end

function get_diff_forw(M, N)
    return x -> diff2d_forw(reshape(x, M, N))
end

function get_diff_adj(M, N)
    return d -> diff2d_adj(d, M, N)
end

diff_map_2d = (M,N) -> LinearMapAA(get_diff_forw(M, N), get_diff_adj(M, N), (N*(M-1) + (N-1)*M, M*N), T=Float32)

# M=30
# N=20
# forw = x -> diff2d_forw(reshape(x,M,N))
# adj = d -> diff2d_adj(d,M,N)
# C = LinearMapAA(forw, adj, (N*(M-1) + (N-1)*M, M*N))
