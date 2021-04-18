% read image(.mat data)
img = load('images/im05_flit01.mat');
k = img.f;
x = img.x;
y = img.y;

% applying Weiner with noise
% sigma: noise var
% signal_var: should be original signal, here using blurred
sigma = 0.02;
signal_var = var(y(:));
NSR = sigma / signal_var;
y_wnr = deconvwnr(y, k, NSR);

% lucy
y_lucy = deconvlucy(y,k,100);

% deconv_outlier -- best performance
% sigma: standard deviation for Gaussian noise (for inlier data)
% reg_str: regularization strength for sparse priors (0.003), 
%          larger means smoother
sigma = 0.02;
reg_str = 0.003;
y_deconv = deconv_outlier(y, k, sigma, reg_str);


figure; imshow(y_deconv)
