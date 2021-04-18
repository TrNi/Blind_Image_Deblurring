"""
This file  provides convolution linear maps created from image or kernel,
with robust sanity checks.

Adopted strategy:
Julia imfilter 'same' and MATLAB conv2('same'):
Both crop central part of the output and return it for 'same' convolution.
Our linear maps produce the exact same output and are checked for 6 sanity checks.
"""
precompile
using LinearMapsAA
using FFTW:fftshift,ifftshift,fft,ifft
using ImageFiltering:parent, padarray,Fill,centered,freqkernel,reflect,imfilter
# using DSP:conv # For sanity checks against full size convolutions.

#support functions:
matshape = (x,M,N) -> reshape(x,M,N)           # converting vectors to matrix.
ffts = X-> fftshift(fft(ifftshift(X)))         # fft with appropriate shits.
iffts = X-> fftshift(ifft(ifftshift(X)))       # ifft with appropriate shits.

"""
Function pad_it(X,padsize) performs zero padding on 2D array X
in a symmetric or near-symmetric way.

In:
X: 2D array of anysize [M,N]
padsize: tuple or 2-element array of preferably odd elements.
padsize[1],padsize[2] represent total rows and columns to be padded to X.

Out:
Xpad: padded array of size [M+padsize[1]-1, N+padsize[2]-1]
"""
function pad_it(X,padsize)

pad1,pad2 = padsize
# Fill(x,(a,b),(c,d)) adds 'a' rows, 'b' columns before x;
# and 'c' rows, 'd' columns after x.
Xpad = padarray(X,
Fill(0,
(Int(floor((pad1-1)/2)),Int(floor((pad2-1)/2))),
(Int(ceil((pad1-1)/2)),Int(ceil((pad2-1)/2)))))
return Xpad
end

"""
getmap_K(K,M,N) produces a convolution linear map Ak from kernel K such that
Ak * vec(X) = vec(conv(K,X)), where conv() represents 2D same convolution.

In:
K: Kernel of odd size [p,q]
M,N: size of an even-sized image

Out:
Linear map (forward and adjoint) of size [M*N,M*N].
"""
function getmap_K(K,M,N)
    # Latent image sizes must be even, kernel sizes must be odd.
    p,q = size(K)
    y1,y2 = [M,N] .+ [p,q] .-1 # This is a vector.
    K_fft1 = fftshift(fft(pad_it(matshape(K,p,q),[M,N])))
    K_fft2 = ffts(pad_it(matshape(K,p,q),[M,N]))
    get_imfft = x -> ffts(pad_it(matshape(x,M,N),[p,q])) # Here M,N, size_k are given, not variable inputs.
    process = X ->  real(iffts(X)) # Here M,N, size_ k are given, not variable inputs.
    ind1 = Int(y1/2)-Int(M/2)
    ind2 = Int(y2/2)-Int(N/2)
    unpad_X1 = Z -> circshift(Z,(-ind1,-ind2))[1:M,1:N]
    ind3 = Int(y1/2 -floor(p/2))
    ind4 = Int(y2/2 -floor(q/2))
    unpad_X2 = Z -> circshift(Z,(-ind3,-ind4))[1:M,1:N]
    forw_map = x -> vec(unpad_X1(process(get_imfft(x) .* K_fft1)))
    adj_map = x -> vec(unpad_X2(process(get_imfft(x) .* conj.(K_fft2))))
    return LinearMapAA(forw_map,adj_map,(M*N,M*N), T=Float32)
end

"""
getmap_X(X,size_k1,size_k2) produces a convolution linear map Ax from image X such that
Ax * vec(K) = vec(conv(X,K)), where conv() represents 2D same convolution.

In:
X: Image of even size [M,N]
size_k1,size_k2: size of an odd-sized kernel

Out:
Linear map (forward and adjoint) of size [M*N,size_k1*size_k2].
"""
function getmap_X(X,size_k1,size_k2)
    # Latent image sizes must be even, kernel sizes must be odd.
    M, N = size(X)
    p,q = [size_k1,size_k2]
    y1,y2 = [M,N] .+ [p,q] .-1 # This is a vector.
    X_fft1 = fftshift(fft(pad_it(X,[p,q])))
    X_fft2 = ffts(pad_it(X,[p,q]))
    pad_k = k -> pad_it(matshape(k,p,q),[M,N])
    process = X ->  real(iffts(X))
    ind1 = Int(y1/2 - M/2)
    ind2 = Int(y2/2 - N/2)
    unpad_X1 = Z -> circshift(Z,(-ind1,-ind2))[1:M,1:N]
    ind3 = Int(y1/2 -floor(p/2))
    ind4 = Int(y2/2 -floor(q/2))
    unpad_k2 = K -> circshift(K,(-ind3,-ind4))[1:p,1:q]
    forw_map = k -> vec(unpad_X1(process(ffts(pad_it(matshape(k,p,q),[M,N])) .* X_fft1)))
    adj_map = y -> vec(unpad_k2(process(ffts(pad_it(matshape(y,M,N),[p,q])) .* conj.(X_fft2))))
    return LinearMapAA(forw_map,adj_map,(M*N,p*q),T=Float32)#; T= ComplexF32)
end



"""
Function sanity_checks() provides 6 sanity checks that make sure that implementation and usage of the
convolutional linear maps above matches with Julia conv() and imfilter() commands, while performing
same-sized convolution.

Sanity Checks:
1. Forward getmap_K
2. Adjoint getmap_K
3. Forward getmap_X
4. Adjoint getmap_X
5. isapprox(Matrix(lm1'), Matrix(lm1)') #lm1 obtained from getmap_K
6. isapprox(Matrix(lm2'), Matrix(lm2)') #lm2 obtained from getmap_X

Additional sanity check that can be added:
y' * (A * x) = (A' * y)' * x
"""
function sanity_checks()
    # Using imfilter for sanity check. imfilter performs 'same' correlation and returns center crop of output.
    # We need to flip kernel before passing in. Default padding is not zero, so we set that.
    imfilter1 = (X,K) -> round.(imfilter(X,reflect(centered(K)),Fill(zero(eltype(X)))),digits=3)
    roundmatreal = (x,m,n) -> round.(reshape(real(x),m,n),digits=3)
    println("") # This line initiates printing.
    function sanity_subfun(X,K,M,N;onlysmallchecks=false)
        p,q = size(K)
        lm_z = getmap_K(K,M,N)
        # Check fwd map 1
        op1 = roundmatreal(lm_z*vec(X),M,N)
        true1 = imfilter1(X,K)
        # Check adj map 1
        flipped_K = (reflect(hcat(zeros(p+1,1),vcat(zeros(1,q),K)))); # first append a row and a column of zeros to K and then reflect!
        op2 = roundmatreal(lm_z' *vec(X),M,N)
        true2 = imfilter1(X,flipped_K)
        # Check fwd map 2
        lm_z2 = getmap_X(X,p,q)
        op3 = roundmatreal(lm_z2*vec(K),M,N)
        true3= true1 # identical conv. op.
        # Check adj map 2
        flipped_X = reflect(hcat(zeros(M+1,1),vcat(zeros(1,N),X))); # first append a row and a column of zeros to X and then reflect!
        op4 = roundmatreal(lm_z2' *vec(X),p,q);
        true4 = roundmatreal((vec(X)' * Matrix(lm_z2))',p,q);
        c1 = isapprox(op1,true1)
        c2 = isapprox(op2,true2)
        c3 = isapprox(op3,true3)
        c4 = isapprox(op4,true4)
        if onlysmallchecks==false
            approxc1 = isapprox(Matrix(lm_z'), Matrix(lm_z)')
            approxc2 = isapprox(Matrix(lm_z2'), Matrix(lm_z2)')
        end
        if c1 && c2 && c3 && c4
            if onlysmallchecks==false
                if !approxc1 || !approxc2
                    println("\n Small Checks Passed, matrix checks failed! ");
                else
                    println("\n All Checks Passed for present case. ");
                end
            else
                println("\n All Checks Passed for present case. ");
            end
        else
            println("\n Incorrect, check this: \n");
            show(stdout,"text/plain",(c1,c2,c3,c4));println("");
            if onlysmallchecks==false
                show(stdout,"text/plain",(approxc1,approxc2));println("");
            end
            # show(stdout,"text/plain", op1); println(""); # To print arrays.
            # show(stdout,"text/plain", true1); println("");
        end
    end
    # Keep image even, kernel odd - mandatory by design.
    X1 = [3 4 10 12;-10 0 9 -3; 1 2 23 6; -40 30 20 35] # 4x4
    M1,N1 = size(X1)
    #1. Kernel: [odd, odd]: 3x3
    K1 = [1 2 3; 2 6 4; 5 8 9]; sanity_subfun(X1,K1,M1,N1)
    #2. Kernel: 3x5
    K2 = [10 20 -3 4 2; -22 44 -16 0 2; 0 7 12 3 4]; sanity_subfun(X1,K2,M1,N1)
    #3. Kernel: 5x3
    K3 = [18 2 1; -6 9 2; 1 -2 9; 4 17 20; 3 -21 43]; sanity_subfun(X1,K3,M1,N1);

    X2 = [-10 0 9 -3; 1 2 23 6] # 2x4
    M2,N2 = size(X2)
    sanity_subfun(X2,K1,M2,N2);
    sanity_subfun(X2,K2,M2,N2);
    sanity_subfun(X2,K3,M2,N2);
    X3 = reshape(range(0,stop=272^2-1),272,272)/(272*272/64);
    K4 = reshape(range(0,stop=21^2 -1),21,21)/42;
    M3,N3 = size(X3);
    sanity_subfun(X3,K4,M3,N3,onlysmallchecks=true);println("");
end
sanity_checks();







"""
Function conv3(X,K) performs convolution between image X and kernel K using
getmap_K() function above. This is just an illustration of using above functions.
"""
function conv3(X, K)
    M, N = size(X)
    lm = getmap_K(K, M, N)
    return matshape(lm*vec(X), M, N)
end


"""
Function getmap_psX(X,size_k) is very similar to getmap_X(X,p,q) above,
except that getmap_psX() can produce circular shifts because X is not padded.
This function is only used in a rough estimate of Lipschitz constant in MM-based methods.
"""
function getmap_psX(X, size_k)
    M, N = size(X)
    p, q = size_k
    pad_k = k -> parent(padarray(k, Fill(0, (Int(ceil((M - p)/2)), Int(ceil((N - q)/2))),
                            (Int(floor((M - p)/2)), Int(floor((N - q)/2))))))
    unpad_k = k -> k[Int(ceil((M - p)/2))+1 : Int(ceil((M - p)/2))+p, Int(ceil((N - q)/2))+1 : Int(ceil((N - q)/2))+q]
    fft_X = fft2(X)
    A = LinearMapAA(x -> real(vec(unpad_k(ifft2(fft2(pad_k(reshape(x, p, q))) .* abs2.(fft_X))))),
                    y -> real(vec(unpad_k(ifft2(fft2(pad_k(reshape(y, p, q))) .* abs2.(fft_X))))),
                    (p*q,p*q); T= Float64)
    return A
end
