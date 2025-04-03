import torch as t
from torch.fft import fft2, fftshift, ifft2
from torchvision.transforms.v2 import Transform


def _applWindow2D(in_img, window=None):
    height, width = in_img.shape[-2], in_img.shape[-1]

    if window is None:
        return in_img
    elif window == "Hanning":
        window_1d = t.hann_window(height)
    elif window == "Hamming":
        window_1d = t.hamming_window(height)
    elif window == "Bartlett":
        window_1d = t.bartlett_window(height)
    elif window == "Blackman":
        window_1d = t.blackman_window(height)
    else:
        raise NotImplementedError("Windowing function %s is not implemented" % window)

    window_2d = t.outer(window_1d, window_1d)
    window_2d /= t.sum(window_2d)

    result = in_img * window_2d.unsqueeze(0).unsqueeze(0)

    return result


def _azimuthalAverage(in_amp, center=None):
    img_h, img_w = in_amp.shape[-2], in_amp.shape[-1]
    y, x = t.meshgrid(t.arange(img_h), t.arange(img_w), indexing="ij")

    if center is None:
        center = t.tensor([(x.max() - x.min()) / 2.0, (y.max() - y.min()) / 2.0])

    r = t.hypot(x - center[0], y - center[1])
    ind = t.argsort(r.flatten())
    r_sorted = r.flatten()[ind]
    i_sorted = in_amp.flatten(start_dim=1, end_dim=-1)[:, ind]
    r_int = r_sorted.int()
    delta = r_int[1:] - r_int[:-1]  # assumes all radii represented
    rind = t.where(delta)[0]  # location of changed radius
    nr = rind[1:] - rind[:-1]  # number of radius bin

    # cumulative sum to figure out sums for each radius bin
    csim = t.cumsum(i_sorted, dim=-1)
    tbin = csim[:, rind[1:]] - csim[:, rind[:-1]]
    radial_prof = tbin / nr.unsqueeze(0)

    return radial_prof


def calDensity(in_data, window="Hanning", normalize=True):
    # convert to grayscale image (w/o weighted average)
    in_data_gray = t.mean(in_data.cpu(), dim=1, keepdim=True)

    # apply windowing
    processed_data = _applWindow2D(in_data_gray, window=window)

    # convert to spectrum
    fspec = fft2(processed_data)
    f_shifted = fftshift(fspec)

    # compute amplitude
    amp = t.real(t.sqrt(f_shifted * t.conj(f_shifted)))

    # compute PSD
    psd = _azimuthalAverage(amp)

    # normalize if needed
    if normalize:
        psd_min, _ = t.min(psd, dim=-1, keepdim=True)
        psd_max, _ = t.max(psd, dim=-1, keepdim=True)
        psd = (psd - psd_min) / (psd_max - psd_min + 1e-8)

    return psd


class ImgWhitening(Transform):
    def __init__(self, window="Hanning"):
        super().__init__()
        self.window = window

    def forward(self, in_img):
        in_shape = in_img.shape
        if len(in_shape) == 3:
            in_img = in_img.unsqueeze(0)
            out_shape = [in_shape[-3], in_shape[-2], in_shape[-1]]
        else:
            out_shape = [-1, in_shape[-3], in_shape[-2], in_shape[-1]]

        # get mean and variance of the original image
        org_mean = t.mean(in_img, dim=(-2, -1), keepdim=True)
        org_var = t.var(in_img - org_mean, dim=(-2, -1), keepdim=True)

        # apply windowing
        processed_img = _applWindow2D(in_img, window=self.window)

        # convert to amplitude spectrum
        fspec = fft2(processed_img)
        img_amp = t.real(t.sqrt(fspec * t.conj(fspec)))

        # whitening
        whitened_img = t.real(ifft2(fspec / (img_amp + 1e-8)))

        # restore mean and variance
        cur_mean = t.mean(whitened_img, dim=(-2, -1), keepdim=True)
        cur_var = t.var(whitened_img - cur_mean, dim=(-2, -1), keepdim=True)

        out_img = (whitened_img - cur_mean) * t.sqrt(org_var / (cur_var + 1e-8)) + org_mean
        out_img = out_img.view(out_shape)

        return out_img
