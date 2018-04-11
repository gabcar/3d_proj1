import numpy as np
from scipy.stats import multivariate_normal
from matplotlib import pyplot as plt
from T2 import dwt2, idwt2
import pywt
from cv2 import imread, resize


def generateImage(_dim, _res):
    out_img = np.zeros((_res,_res))

    out_img[int(_res/2), int(_res/2)] = 1

    return out_img

def generateWavelets(lp_d, hp_d, _res=100, levels=2):

    padding = levels * 2 * len(lp_d)

    posx = int(np.round(1.5*(_res + 2 * padding) / (2 ** levels)))
    posy = int(np.round(.5*(_res + 2 * padding) / (2 ** levels)))

    im = np.zeros((_res + 2 * padding,_res + 2 * padding))
    im[posy, posx] = 1

    return im

def generate3D(_res, _slices, im=None):

    pos = int(np.round(_res/2))-1
    posz = int(np.ceil(_slices/2))-1

    if im is None:
        vol = np.zeros((_res, _res, _slices))
        vol[pos, pos, posz] = 1
    else:
        vol = np.zeros((im.shape[0], im.shape[1], _slices))
        vol[:, :, posz] = im
    return vol

def readImage():
    return imread('t1_flair_2d.py')

def randomUndersample(_img, _p, _order=None):
    if _order is None:
        img_flat = _img.flatten()
        img_out = np.zeros(img_flat.shape)
        img_out = np.array(img_out, dtype=complex)

        p = 1-_p

        n = len(img_flat)

        n_points = np.int(np.round((p*n)))
        pix = np.random.choice(a=len(img_flat), size=n_points, replace=False)
        img_out[pix] = img_flat[pix]
        return img_out.reshape(int(np.sqrt(n)),int(np.sqrt(n)))

    elif len(_order.shape) == 1:
        img_out = np.array(_img, dtype=complex)
        img_out[_order, :] = _img[_order, :]
        return img_out

def fig_4_a():
    img = generateImage([-2,2], 100)
    fft_img = np.fft.fft2(img)
    fft_undersample = randomUndersample(fft_img, 0.5)

    im_recon = np.real(np.fft.ifft2(fft_undersample))

    plt.imshow(im_recon)
    plt.show()

def fig_4_b():
    wl_name = 'haar'
    lp_d, hp_d, lp_r, hp_r = map(np.array, pywt.Wavelet(wl_name).filter_bank)
    num_levels = 2

    wv_img1 = generateWavelets(lp_d, hp_d, levels=num_levels)

    # step 1: IDWT of wavelet domain
    iwv_img = idwt2(wv_img1, lp_r, hp_r, levels=num_levels)

    # step 2: FFT of image domain
    fft_img = np.fft.fft2(iwv_img)

    # step 3: random undersampling of FFT domain
    us_fft = randomUndersample(fft_img, 0.2)

    # step 4: IFFT of undersampled FFT domain
    ifft_img = np.real(np.fft.ifft2(us_fft))

    # step 5: DWT of image domain
    wv_img2 = dwt2(ifft_img, lp_d, hp_d, levels=num_levels)

    fig, ax = plt.subplots(2,3)
    plt.subplot(2,3,1)
    plt.imshow(wv_img1)

    plt.subplot(2,3,2)
    plt.imshow(iwv_img)

    plt.subplot(2,3,3)
    plt.imshow(np.abs(np.fft.fftshift(fft_img)))

    plt.subplot(2,3,6)
    plt.imshow(np.abs(np.fft.fftshift(us_fft)))

    plt.subplot(2,3,5)
    plt.imshow(ifft_img)

    plt.subplot(2,3,4)
    plt.imshow(wv_img2)

    plt.show()

def fig_5():
    wl_name = 'haar'
    lp_d, hp_d, lp_r, hp_r = map(np.array, pywt.Wavelet(wl_name).filter_bank)
    num_levels = 5
    padding = num_levels * 2 * len(lp_d)
    res = 2048
    slices = 5
    dft = generateWavelets(lp_d, hp_d, _res=res)
    dft_vol = generate3D(_res=res, _slices=slices, im=dft)
    img_vol = np.zeros((res, res, slices))
    fft_vol = np.array(np.zeros(img_vol.shape), dtype=complex)
    us_fft_vol = np.copy(fft_vol)
    ifft_vol = np.copy(img_vol)
    dft_vol2 = np.copy(img_vol)

    p = 1 - 0.5
    s = int(np.round(p*res))

    #fig 5.a
    order = np.random.choice(res, size=s, replace=False)
    for slice in range(slices):
        img_vol[:, :, slice] = idwt2(dft_vol[:, :, slice],lp_r, hp_r, levels=num_levels)[padding:padding+img_vol.shape[0], padding:padding+img_vol.shape[1]]
        fft_vol[:, :, slice] = np.fft.fft2(img_vol[:, :, slice])  # fft
        us_fft_vol[order, :, slice] = fft_vol[order, :, slice]  # undersampled fft
        ifft_vol[:, :, slice] = np.abs(np.fft.ifft2(us_fft_vol[:, :, slice]))  # inverse of undersampled fft
        max_intensity = np.max(np.abs(fft_vol[order, :, slice]))
        min_intensity = np.min(np.abs(fft_vol[order, :, slice]))
        dft_vol2[:, :, slice] = dwt2(ifft_vol[:, :, slice], lp_d, hp_d, levels=num_levels)  # dft of usampled imdomain


        fig, ax = plt.subplots(2, 3)
        plt.subplot(2,3,1)
        plt.imshow(dft_vol[:,:,slice])

        plt.subplot(2,3,2)
        plt.imshow(img_vol[:,:,slice])

        plt.subplot(2,3,3)
        plt.imshow(np.abs(np.fft.fftshift(fft_vol[:,:,slice])))

        plt.subplot(2,3,6)
        plt.imshow(np.abs(np.fft.fftshift(us_fft_vol[:,:,slice])), vmin=min_intensity, vmax=max_intensity)

        plt.subplot(2,3,5)
        plt.imshow(ifft_vol[:,:,slice])

        plt.subplot(2,3,4)
        plt.imshow(dft_vol2[:,:,slice])

        fig.savefig("plots/fig5_a_slice_{}_of_{}".format(slice+1,slices))



if __name__ == '__main__':
    #fig_4_a()
    #fig_4_b()
    fig_5()
