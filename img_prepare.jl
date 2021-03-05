using Images, FileIO
using JLD
using Colors
using MAT

workdir = pwd()

# Sun
img_path = "./Data/Sun/kernels/"
cd(img_path)
filenames = readdir()
for filename in filenames
    img = Float64.(Gray.(load(filename)))
    imagepath = splitext(filename)[1]*".jld"
    save(imagepath,"kernel",img)
end
cd(workdir)

img_path = "./Data/Sun/ground_truth_80/"
cd(img_path)
filenames = readdir()
for filename in filenames
    img = Float64.(Gray.(load(filename)))
    imagepath = "./groundtruth80/"*splitext(filename)[1]*".jld"
    save(imagepath,"gray",img)
end
cd(workdir)

# Dong
img_path = "./Data/Dong/"
cd(img_path)
filenames = readdir()
for filename in filenames
    img = load(filename)
    img = channelview(img)
    r = Float64.(img[1,:,:])
    g = Float64.(img[2,:,:])
    b = Float64.(img[3,:,:])
    imagepath = splitext(filename)[1] * ".jld"
    save(imagepath,"red",r,"green",g,"blue",b)
end
cd(workdir)


# Levin
img_path = "./Data/Levin/"
cd(img_path)
filenames = readdir()
for filename in filenames
    data = matread(filename)
    kernel = data["f"]
    truth = data["x"]
    blurred = data["y"]
    blurname = splitext(filename)[1]
    imagename = rsplit(blurname, "_")[1]
    kernelname = rsplit(blurname, "_")[2]
    save("./blurred/"*blurname*".jld","gray",blurred)
    save("./truth/"*imagename*".jld","gray",truth)
    save("./kernel/"*kernelname*".jld","kernel",kernel)
end
cd(workdir)
