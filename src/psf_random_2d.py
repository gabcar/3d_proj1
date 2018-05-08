import numpy as np
from matplotlib import pyplot as plt
from T2 import dwt2, idwt2
import pywt

def plot3D(img1, title=''):
    fig = plt.figure(figsize=plt.figaspect(1))
    res = img1.shape[0]
    X = np.arange(0, res)
    Y = np.arange(0, res)
    X, Y = np.meshgrid(X, Y)

    plt.title(title)
    img1 = img1/np.max(img1)
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    surf = ax.plot_surface(X, Y, img1, color='r')
    #ax.set_zlim(-0.01, 1.01)

    plt.tight_layout()
    fig.savefig(title)


def generateImage(_res):
    out_img = np.zeros((_res,_res))

    out_img[int(_res/2), int(_res/2)] = 1

    return out_img

def generateWavelets(lp_d, hp_d, _res=100, levels=2, fine=0, padding=0):
    img = generateImage(_res)
    wv = dwt2(img, lp_d, hp_d, levels)
    z = np.zeros((_res + padding,_res + padding))
    if fine == 0:
        p = int(_res/4)
        z[:p, :p] = wv[:p, :p]
    else:
        p = int(_res/2)
        z[p:, p:] = wv[p:, p:]
    z = z/np.max(z)
    return z

def generate3D(_res, _slices, _im):
    vol = np.zeros((_im.shape[0], _im.shape[1], _slices))
    vol[int(_res/2), :, :] = _im
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

    im_rec = np.real(np.fft.ifft2(fft_undersample))

    # Make data.
    X = np.arange(0, 116)
    Y = np.arange(0, 116)
    X, Y = np.meshgrid(X, Y)


    fig = plt.figure(figsize=plt.figaspect(1))

    ax = fig.add_subplot(2, 2, 1, projection='3d')
    surf = ax.plot_surface(X, Y, img, cmap='Reds')

    ax = fig.add_subplot(2, 2, 2, projection='3d')
    surf = ax.plot_surface(X, Y, np.real(fft_img), cmap='Reds')

    ax = fig.add_subplot(2, 2, 3, projection='3d')
    surf = ax.plot_surface(X, Y, im_rec, cmap='Reds')

    ax = fig.add_subplot(2, 2, 4, projection='3d')
    surf = ax.plot_surface(X, Y, np.real(fft_undersample), cmap='Reds')

    #plt.show()

def fig_4_b():
    wl_name = 'haar'
    lp_d, hp_d, lp_r, hp_r = map(np.array, pywt.Wavelet(wl_name).filter_bank)
    num_levels = 2

    wv_img1 = generateWavelets(lp_d, hp_d, levels=num_levels, fine=1)

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

    fig = plt.figure(figsize=plt.figaspect(2/3))

    X = np.arange(0, 100)
    Y = np.arange(0, 100)
    X, Y = np.meshgrid(X, Y)

    print(wv_img1.shape)
    ax = fig.add_subplot(2, 3, 1, projection='3d')
    surf = ax.plot_surface(X, Y, wv_img1, color='r')

    ax = fig.add_subplot(2, 3, 2, projection='3d')
    surf = ax.plot_surface(X, Y, iwv_img, color='r')

    ax = fig.add_subplot(2, 3, 3, projection='3d')
    surf = ax.plot_surface(X, Y, np.real(np.fft.fftshift(fft_img)), color='r')

    ax = fig.add_subplot(2, 3, 6, projection='3d')
    surf = ax.plot_surface(X, Y, np.real(np.fft.fftshift(us_fft)), color='r')

    ax = fig.add_subplot(2, 3, 5, projection='3d')
    surf = ax.plot_surface(X, Y, ifft_img, color='r')

    ax = fig.add_subplot(2, 3, 4, projection='3d')
    surf = ax.plot_surface(X, Y, wv_img2, color='r')

    fig.tight_layout

    #plt.show()

def fig_5a():
    wl_name = 'haar'
    lp_d, hp_d, lp_r, hp_r = map(np.array, pywt.Wavelet(wl_name).filter_bank)
    num_levels = 2
    res = 32
    slice = int(res/2)
    p = 0.5

    mask = np.random.choice([0, 1], size=(res,), p=[1-p, p])
    mask = np.tile(mask, (res, 1)).T

    wavelet_slice = generateWavelets(lp_d, hp_d, res, fine=1)
    im_vol = np.zeros((res, res, res))
    im_vol[slice, :, :] = idwt2(wavelet_slice, lp_r, hp_r, num_levels)

    fft_vol = np.fft.fftn(im_vol, axes=(1,0))

    us_fft_vol = np.copy(fft_vol)*0
    us_fft_vol[slice, :, :] = np.multiply(fft_vol[slice, :, :], mask)

    ifft_vol = np.fft.ifftn(us_fft_vol, axes=(1,0))
    dwt_slice = dwt2(np.real(ifft_vol[slice, :, :]), lp_d, hp_d, num_levels)

    plot3D(np.abs(dwt_slice), "Single slice")

def fig_5b():
    wl_name = 'haar'
    lp_d, hp_d, lp_r, hp_r = map(np.array, pywt.Wavelet(wl_name).filter_bank)
    num_levels = 2
    res = 32
    slice = int(res/2)
    p = 0.5

    wavelet_slice = generateWavelets(lp_d, hp_d, res, fine=1)
    im_vol = np.zeros((res, res, res))
    im_vol[slice, :, :] = idwt2(wavelet_slice, lp_r, hp_r, num_levels)

    fft_vol = np.fft.fftn(im_vol, axes=(1,0))
    mask = np.random.choice([0, 1], size=(res, res), p=[1-p, p])
    us_fft_vol = np.copy(fft_vol)*0
    us_fft_vol[slice, :, :] = np.multiply(fft_vol[slice, :, :], mask)
    ifft_vol = np.fft.ifftn(us_fft_vol, axes=(1,0))
    dwt_slice = dwt2(np.real(ifft_vol[slice, :, :]), lp_d, hp_d, num_levels)

    plot3D(np.abs(dwt_slice), "Multislice")

def fig_5c():
    wl_name = 'haar'
    lp_d, hp_d, lp_r, hp_r = map(np.array, pywt.Wavelet(wl_name).filter_bank)
    num_levels = 2
    res = 32
    slice = int(res/2)
    p = 0.5

    wavelet_slice = generateWavelets(lp_d, hp_d, res, fine=1)
    im_vol = np.zeros((res, res, res))
    im_vol[slice, :, :] = idwt2(wavelet_slice, lp_r, hp_r, num_levels)

    fft_vol = np.fft.fftn(im_vol, axes=(0,1,2))
    mask = np.random.choice([0, 1], size=(res, res), p=[1-p, p])

    us_fft_vol = np.copy(fft_vol)*0

    print(mask.shape)
    for i in range(res):
        us_fft_vol[i, :, :] = np.multiply(fft_vol[i, :, :], mask)

    ifft_vol = np.fft.ifftn(us_fft_vol, axes=(0,1,2))

    dwt_slice = dwt2(np.real(ifft_vol[slice, :, :]), lp_d, hp_d, num_levels)

    plot3D(np.abs(dwt_slice), title="3D")


if __name__ == '__main__':
    #fig_4_a()
    #fig_4_b()
    fig_5a()
    plt.show()
    fig_5b()
    plt.show()
    fig_5c()
    plt.show()
