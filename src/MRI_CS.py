"""
Provide Python code to perform the experiments in demo4.m. Investigate the effect of using different sampling patterns in k-space, i.e. vary used pdf, as well as modifying the thresholding approach

Step 1:
Display the mask and the PDF

Step 2:
Compute the 2DFourier transform of the image. Multiply with the mask, divide
by the PDF. Compute the inverse Fourier transform and display the result.

Step 3:
Implement the POCS algorithm for 2D images. Use lambda value from the
thresholding experiment. Use 20 iterations.
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from scipy.io import loadmat
from cv2 import imread
from cv2 import resize
import pywt
from T2 import dwt2, idwt2


def displayMaskAndPDF(_mask, _pdf):
    """
    in: _mask: 2D ndarray (h x w)
        _pdf:  2D ndarray (h x w)
    """

    plt.subplots(1, 2)

    plt.subplot(1, 2, 1)
    plt.imshow(_mask, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.imshow(_pdf, cmap='gray')

    plt.show()
    return 1


def loadMat(path="workshop/brain.mat"):
    mat = loadmat('src/workshop/brain.mat')
    brain = mat['im']
    mask = mat['mask_vardens']
    pdf = mat['pdf_vardens']
    return brain, mask, pdf


def generateNormalPdf(_shape=10, _range=[-2, 2], _mu=[0, 0], _sigma=[[.25, 0], [0, 1]], _res=[1024, 1024]):
    """
    in:    ndarray (h,w)
    range: ndarray range of pdf
    """
    X = np.linspace(_range[0], _range[1], _res[0])
    Y = np.linspace(_range[0], _range[1], _res[1])
    pos = np.zeros((_res[0], _res[1], 2))
    pos[:, :, 0], pos[:, :, 1] = np.meshgrid(X, Y)
    normal = multivariate_normal(_mu, _sigma).pdf(pos)

    return normal


def generateUniformPDF(_res=(10, 10)):
    return np.ones(_res) / (_res[0] * _res[1])


def generateMask(_pdf, _p=0.3):
    """
    in:     _im:    ndarray, image (gray scale)
            _pdf:   ndarray, same size as _im
            _p:     fraction of probability distribution to sample
    """

    pdf = _pdf / np.abs(_pdf).sum()  # normalize
    flat_pdf = np.ravel(pdf)
    idx = np.arange(len(flat_pdf))
    n = int(np.round(len(flat_pdf) * _p))
    idx_mask = np.random.choice(idx, size=n, p=flat_pdf, replace=False)

    mask = np.zeros((len(flat_pdf,)))

    mask[idx_mask] = 1

    return mask.reshape(_pdf.shape)


def generateMaskFromPng(_path="db.png", _shape=(1024, 1024)):
    """
    generates binary mask from .png file

    in:     _path: path to png      str
            _shape: shape of image  tuple
    """
    im = imread(_path, 0)  # read image in grayscale
    im = im / np.max(im)   # rescale
    im = (im > 0.5) * 1    # binary 1/0

    return resize(im, _shape)


def FFT_special(_im, _mask, _pdf):
    """
    Computes the 2DFourier transform of the image. Multiplies with the mask, divides
    by the PDF. Computes the inverse Fourier transform and returns the result.
    """
    fft_im = np.fft.fftshift(np.fft.fft2(_im))
    ifft_im = np.multiply(_mask, fft_im) / _pdf
    return np.fft.ifft2(np.fft.fftshift(ifft_im))


def fft2c(_im):
    """
    fctr = size(x,1)*size(x,2);
    for n=1:size(x,3)
        res(:,:,n) = 1/sqrt(fctr)*fftshift(fft2(ifftshift(x(:,:,n))));
    end
    """
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(_im)))


def ifft2c(_im):
    """
    fctr = size(x,1)*size(x,2);
    for n=1:size(x,3)
        res(:,:,n) = sqrt(fctr)*fftshift(ifft2(ifftshift(x(:,:,n))));
    end
    """
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(_im)))


def idwt2c(_im, num_levels=1):
    wl_name = 'db1'
    lp_d, hp_d, lp_r, hp_r = map(np.array, pywt.Wavelet(wl_name).filter_bank)
    return idwt2(_im, lp_r, hp_r, num_levels)


def dwt2c(_im, num_levels=1):
    wl_name = 'db1'
    lp_d, hp_d, lp_r, hp_r = map(np.array, pywt.Wavelet(wl_name).filter_bank)
    return dwt2(np.abs(_im), lp_d, hp_d, num_levels)


def softThresh(x, lambd):
    print(np.max(np.abs(x)))
    # y = (abs(x) > lambda).*(x.*(abs(x)-lambda)./(abs(x)+eps));
    a = np.abs(x) > lambd
    b = np.multiply(x, np.abs(x) - lambd)
    b = np.divide(b, abs(x) + np.finfo(float).eps)
    y = np.multiply(a, b)
    return y


def hardThresh(x, lambd):
    print(np.max(np.abs(x)))
    y = np.abs(x) > lambd
    x = np.multiply(x, y)
    return x


def POCS(_im, _mask, _pdf, iter=16):
    """
    Implement the POCS algorithm for 2D images. Use lambda value from the
    thresholding experiment. Use 20 iterations.
    """

    # DATA = fft2c(im).*mask_vardens;
    DATA = np.multiply(fft2c(_im), _mask)
    # im_cs = ifft2c(DATA./pdf_vardens); % initial value
    im_cs = ifft2c(np.divide(DATA, _pdf))

    #fig = plt.figure()
    for i in range(iter):
        #im_cs = W'*(SoftThresh(W*im_cs,0.025));
        im_cs = dwt2c(im_cs)
        im_cs = idwt2c(hardThresh(im_cs, 0.025))
        #im_cs = ifft2c(fft2c(im_cs).*(1-mask_vardens) + DATA);
        im_cs = np.multiply(fft2c(im_cs), (1 - _mask))
        im_cs += DATA
        im_cs = ifft2c(im_cs)

        #plt.imshow(np.abs(im_cs), cmap='gray')
        #plt.title("Iteration: {}".format(i))
        #plt.pause(0.3)
    return im_cs


def noCsRecon(_im, _mask, _pdf):
    DATA = np.multiply(fft2c(_im), _mask)
    recon_image = ifft2c(DATA)
    return recon_image


if __name__ == '__main__':
    #step 1
    brain, mask, pdf = loadMat("db.png")
    #pdf = generateNormalPdf(_res=brain.shape, _sigma=[[.5, 0], [0, .5]])
    pdf_uni = generateUniformPDF(_res=brain.shape)
    mask_uni = generateMask(pdf_uni, _p=.35)
    p = .35
    #displayMaskAndPDF(mask, pdf)

    #step 2

    """
    plt.subplots(1,2)
    plt.subplot(1,2,1)
    plt.imshow(np.abs(FFT_special(brain, mask, pdf)), cmap="gray")
    plt.subplot(1,2,2)
    plt.imshow(np.abs(brain), cmap="gray")
    plt.show()
    """

    # step 3
    im_out = POCS(brain, mask, pdf, iter=20)
    #no_cs = noCsRecon(brain, mask_uni, pdf_uni)

    fig, ax = plt.subplots(1, 4, figsize=(10, 4))
    plt.subplot(1, 4, 1)
    plt.title("Before")
    plt.imshow(np.abs(brain), cmap="gray")

    plt.subplot(1, 4, 2)
    plt.title("After")
    plt.imshow(np.abs(im_out), cmap="gray")

    plt.subplot(1, 4, 3)
    plt.title("pdf")
    plt.imshow(np.abs(pdf), cmap="gray")

    plt.subplot(1, 4, 4)
    plt.title("Mask")
    plt.imshow(np.abs(mask), cmap="gray")

    plt.suptitle("Vardens mask - Hard Threshold - {}% sampling".format(int(p * 100)))
    plt.tight_layout()
    fig.savefig("vardens_mask_hard_threshold_{}.png".format(int(p * 100)))
    plt.show()
