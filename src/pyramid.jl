"""
This file provides utility functions to run alternating minimization
at different scales of images / kernels. Some functions are translated into Julia from
Pan et al.'s MATLAB code, some other functions are written by us.
Some functions are not used in our project.
"""

using ImageFiltering
using Interpolations
"""
This is a Julia version of shock_filter utility available in MATLAB at
https://www.mathworks.com/matlabcentral/fileexchange/29593-shock-filtering-in-matlab
Check MATLAB function description for an easy explanation of parameters and usage.
"""
function shock_filter(img; sigma=0.5, iterations=30, dt=0.25)
    k = Kernel.gaussian(sigma)
    LoG = ImageFiltering.Kernel.Laplacian()
    G = img
    for i = 1:iterations
        G = imfilter(img, k)
        gx, gy = imgradients(G, KernelFactors.sobel)
        normxy = sqrt.(gx.^2 + gy.^2)
        s = -sign.(imfilter(G, LoG))
        G = G .+ dt * s.* normxy
    end
    return G
end

"""
Function init_kernel(minsize) produces a square kernel of
size [minsize,minsize] with uniform values summing to 1.

In:
minsize: size in one dimension, must be >=3.
"""
function init_kernel(minsize)
    k = zeros(minsize, minsize)
    halfid = Int(floor((minsize - 1)/2))
    k[halfid:halfid+2, halfid:halfid+2] .= 1/9 #3x3, 1/9
    return k
end


"""
This function downsamples given image by ratio ret4.
First the image is convolved with a Gaussian filter and then downsampled.
"""
function downSmpImC(img1, ret4)
    if ret4 == 1
        return img1
    end

    sig = (1/pi) * ret4
    g0 = (-50:50).*2*pi
    sf = exp.(-0.5*g0.^2*sig^2)
    sf = sf/sum(sf)
    csf = cumsum(sf)
    csf = min.(csf, csf[end:-1:1])

    ii = csf.>0.05
    sf = sf[ii]
    kernf = kernelfactors((sf, sf))
    img1 = imfilter(img1, kernelfactors((sf, sf)))
    # imfilter produces cross-correlation of size 'same'.
    # Check imfilter1() in sanity_checks() in linearmap_conv.jl.
    M, N = size(img1)
    im_func = CubicSplineInterpolation((1:M, 1:N), img1, extrapolation_bc = 0)
    return max.(0, im_func.(1:1/ret4:M, (1:1/ret4:N)'))
end



"""
Function pyramid_zl(k_size,img_size) provides
a list of different kernel and image sizes to form a pyramid.

In:
* k_size = kernel size in one dimension (square kernel)
* img_size = image size in one dimension (square image)

Out:
ker_list = list of integer kernel sizes in one dimension
img_list = list of integer image sizes in one dimension
"""
function pyramid_zl(k_size, img_size)
    # Input is the kernel size, return an array of ker/img size in the pyramid
    scale_k = 1.5
    scale_im = 1.5
    min_k = 3
    num = min(6, Int(ceil(log(k_size/min_k)/log(scale_k))))
    ker_list = zeros(Int, num)
    img_list = zeros(Int, num)
    for i = 0:num-1
        k = make_odd_round(k_size / (scale_k)^i)
        img = make_even_round(img_size / (scale_im)^i)
        ker_list[i+1] = Int(k)
        img_list[i+1] = Int(img)
    end
    return reverse(ker_list), reverse(img_list)
end


function make_even_round(x)
    x = round(x/2) * 2
    return mod(x, 2) == 0 ? x : x - 1
end

function make_odd_round(x)
    x = round((x+1)/2) * 2
    return mod(x, 2) == 1 ? x : x - 1
end

"""
Function resizeKer_zl(k,k1) resizes given kernel k to new size k1 using replicated padding.
"""
function resizeKer_zl(k, k1; regfunc = :sum)
    k_size = size(k, 1)
    x, y = Tuple(argmax(k))
    d = Int(floor(k1/2))
    k_pad = padarray(k,Pad(:replicate,d, d))
    kout = k_pad[x-d:x+d, y-d:y+d]
    if regfunc === :sum
        kout = kout / sum(kout)
    elseif regfunc === :norm
        kout = kout / norm(kout, 2)
    else
        throw("unknown regfunc")
    end
end

"""
Function find_ker(k,kt) removes shift ambiguity
from kernel k with respect to true kernel kt and
normalizes the final output according to regfunc.

In:
k: kernel which might have some shift w.r.t. reference
kt: true kernel or reference
regfunc: normalization scheme, default: sum to one.

Out:
Shifted k such that k and kt are in maximum alignment,
measured by minimum NRMSE.
"""
function find_ker(k, kt; regfunc = :sum)
      nrmse_min = 1
      i_select = 0
      j_select = 0
      k_best = k
      nrmse_k = k -> norm(k - kt) / norm(kt)
      for i = -11:11
            for j = -11:11
                  k_shift = shift_ker(k, i, j)
                  if regfunc === :sum
					  k_shift = k_shift / sum(k_shift)
				  elseif regfunc === :norm
					  k_shift = k_shift / norm(k_shift, 2)
				  else
					  throw("unknown regfunc")
				  end
                  nrmse_k_shift = nrmse_k(k_shift)
                  if nrmse_k_shift < nrmse_min
                        nrmse_min = nrmse_k_shift
                        i_select = i
                        j_select = j
                        k_best = k_shift
                  end
            end
      end
      return k_best
end

"""
shift_ker(k,x,y) shifts the kernel k by (x,y) pixels in 2D.
"""
function shift_ker(k, x, y)
      if x ≥ 0
            # shift right
            k_pad_x = vcat(zeros(x, size(k, 2)), k)
            k_pad_x = k_pad_x[1:size(k,1),:]
      else
            # shift left
            k_pad_x = vcat(k, zeros(-x, size(k, 2)))
            k_pad_x = k_pad_x[end-size(k,1)+1:end,:]
      end
      if y ≥ 0
            # shift up
            k_pad_xy = hcat(k_pad_x, zeros(size(k, 1), y))
            k_pad_xy = k_pad_xy[:, end-size(k,2)+1:end]
      else
            # shift down
            k_pad_xy = hcat(zeros(size(k, 1), -y), k_pad_x)
            k_pad_xy = k_pad_xy[:, 1:size(k,2)]
      end
      return k_pad_xy
end

"""
Function resizeKer(k,k1) resizes given kernel k to new size k1 using imresize function.
"""
function resizeKer(k, k1)
    # not using ret3,k2, fixsize.
    oldsize,_ = size(k)
    newsize =k1
    # interpolation results are not good, compare 11x11 central interpolated part with imresize
    # k_intp = CubicSplineInterpolation((1:oldsize, 1:oldsize), k, extrapolation_bc = 0)
    # newk1 = round.(k_intp(1:oldsize/(newsize+1):oldsize,1:oldsize/(newsize+1):oldsize),digits=6)
    newk2 = imresize(k,(newsize,newsize)) # to avoid size ambiguity.
    newk = max.(newk2, 0);
    if maximum(newk) > 0
        newk = newk/sum(newk)
    end
    return newk
end

"""
init_kernel_old is an old function, imported from MATLAB code, not utilized in our project.
"""
function init_kernel_old(minsize)
    k = zeros(minsize, minsize)
    k[Int(floor((minsize - 1)/2)), Int(floor((minsize - 1)/2)):Int(floor((minsize-1)/2)+1)] .= 1/2 # not a kronecker delta.
    return k
end


"""
Function resizeKer_old() resizes given kernel in two steps,
first resizing it by ratio ret3, and then converting it to a fixed size k1xk2.

This function is translated directly from Pan et al.'s MATLAB code, please
refer to that documentation for further insights. We do not utilize this function.
"""
function resizeKer_old(k, ret3, k1, k2)
    k = imresize(img, ret3)
    k = max.(k, 0);
    k = fixsize(k, k1, k2)
    if maximum(k) > 0
        k = k/sum(k)
    end
    return k
end
"""
Function fixsize(f,nk1,nk2) is a sub-function of above resizeKer_old() function.
It transforms given kernel f to a new size nk1xnk2 by appropriate zero padding.

This function is translated directly from Pan et al.'s MATLAB code, please
refer to that documentation for further insights. We do not utilize this function.
"""
function fixsize(f, nk1, nk2)
    k1, k2 = size(f) # present size
    while (k1 != nk1) || (k2 != nk2)
        if (k1 > nk1)
            s = sum(f, dims=2)
            if s[1] < s[end]
                f = f[2:end, :]
            else
                f = f[1:end-1, :]
            end
        end
        if (k1 < nk1)
            s = sum(f, dims=2)
            tf = zeros(k1+1, size(f, 2))
            if s[1] < s[end]
                tf[1:k1, :] = f
            else
                tf[2:k1+1, :] = f
            end
            f = tf
        end
        if (k2 > nk2)
            s = sum(f, dims=1)
            if s[1] < s[end]
                f = f[:, 2:end]
            else
                f = f[:, 1:end-1]
            end
        end
        if (k2 < nk2)
            s = sum(f, dims=1)
            tf = zeros(size(f, 1), k2+1)
            if s[1] < s[end]
                tf[:, 1:k2] = f
            else
                tf[:, 2:k2+1] = f
            end
            f = tf
        end

        k1, k2 = size(f)
    end
    return f
end
