import pickle
import random
import re
from pickle import load

import cv2
import numpy as np
import torch as t
from scipy.interpolate import interp1d
from torch import fft, nn
from torchvision.transforms.v2 import Transform


class HistogramMatching(Transform):
    def __init__(self, pkl_path):
        """
        Initialize the HistogramMatching transform with a target histogram.

        Args:
            pkl_path (str): Path to the .pkl file containing the target histogram data.
        """
        super().__init__()
        # Load the target histogram data from the .pkl file
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
            self.target_histogram = data["avg_histogram"]
            self.bin_edges = data["bin_edges"]

        # Compute the cumulative distribution function (CDF) of the target histogram
        self.target_cdf = np.cumsum(self.target_histogram)
        self.target_cdf /= self.target_cdf[-1]  # Normalize the CDF

    def forward(self, img):
        """
        Apply histogram matching to the input image.

        Args:
            img (torch.Tensor): Input image tensor (C, H, W), normalized to [0, 1].

        Returns:
            torch.Tensor: Image with histogram matched to the target.
        """
        if not isinstance(img, t.Tensor):
            raise TypeError("Input should be a torch Tensor.")

        if img.ndim != 3:
            raise ValueError("Input tensor must have 3 dimensions (C, H, W).")

        # Convert image to numpy for processing
        img_np = img.cpu().numpy()

        # Flatten the image and compute its CDF
        img_flat = img_np.ravel()
        hist, bin_edges = np.histogram(img_flat, bins=256, range=(0, 1))
        img_cdf = np.cumsum(hist)
        img_cdf = img_cdf / img_cdf[-1]  # Normalize the CDF

        # Create a mapping from input CDF to target CDF
        interp_input_cdf = interp1d(bin_edges[:-1], img_cdf, bounds_error=False, fill_value=(0, 1))
        interp_target_cdf = interp1d(self.target_cdf, self.bin_edges[:-1], bounds_error=False, fill_value=(0, 1))

        # Map the input image through the CDF mapping
        valid_mask = img_np >= self.bin_edges[0]  # Pixels below the minimum valid bin
        matched_img = np.zeros_like(img_np)
        matched_img[valid_mask] = interp_target_cdf(interp_input_cdf(img_np[valid_mask]))

        # Set pixels below the minimum valid bin to zero
        matched_img[~valid_mask] = 0

        # Convert back to torch tensor
        matched_img = t.tensor(matched_img, dtype=img.dtype, device=img.device)

        return matched_img


def createMask(h, w, r_low_q1, r_high_q1, r_low_q2, r_high_q2):
    assert 0 <= r_low_q1 < r_high_q1 <= 0.5, "Invalid r_low_q1 and r_high_q1."
    assert 0 <= r_low_q2 < r_high_q2 <= 0.5, "Invalid r_low_q2 and r_high_q2."

    # band-pass filter params
    yy, xx = t.meshgrid(t.linspace(0.5, -0.5, h), t.linspace(-0.5, 0.5, w), indexing="ij")
    radius = t.sqrt(xx ** 2 + yy ** 2)

    q1 = (xx >= 0) & (yy >= 0)  # first quadrant
    q2 = (xx < 0) & (yy >= 0)  # second quadrant
    q3 = (xx < 0) & (yy < 0)  # third quadrant (mirror of Q1)
    q4 = (xx >= 0) & (yy < 0)  # fourth quadrant (mirror of Q2)

    # create band-pass filters for each quadrant
    mask = t.zeros_like(radius, dtype=t.bool)
    mask |= q1 & ((radius <= r_high_q1) & (radius >= r_low_q1))
    mask |= q2 & ((radius <= r_high_q2) & (radius >= r_low_q2))
    mask |= q3 & ((radius <= r_high_q1) & (radius >= r_low_q1))
    mask |= q4 & ((radius <= r_high_q2) & (radius >= r_low_q2))

    return mask


class BandpassFilter(Transform):
    def __init__(self, r_low_q1, r_high_q1, r_low_q2, r_high_q2):
        super().__init__()
        assert 0 <= r_low_q1 < r_high_q1 <= 0.5, "Invalid r_low_q1 and r_high_q1."
        assert 0 <= r_low_q2 < r_high_q2 <= 0.5, "Invalid r_low_q2 and r_high_q2."

        self.r_low_q1 = r_low_q1
        self.r_high_q1 = r_high_q1
        self.r_low_q2 = r_low_q2
        self.r_high_q2 = r_high_q2

    def forward(self, img):
        # Ensure the input is a torch tensor
        if not isinstance(img, t.Tensor):
            img = t.tensor(img)

        # img has at least 3 dimensions
        assert img.ndim >= 3, "Input tensor must have at least 3 dimensions (C, H, W)."

        # handle batch dimensions
        head_dim = img.shape[:-3]
        C, H, W = img.shape[-3:]
        img = img.view(-1, C, H, W)

        # create a meshgrid for the frequency space
        yy, xx = t.meshgrid(t.linspace(0.5, -0.5, H), t.linspace(-0.5, 0.5, W), indexing="ij")
        radius = t.sqrt(xx ** 2 + yy ** 2)

        # determine quadrants (normalized indices)
        q1 = (xx >= 0) & (yy >= 0)  # first quadrant
        q2 = (xx < 0) & (yy >= 0)  # second quadrant
        q3 = (xx < 0) & (yy < 0)  # third quadrant (mirror of Q1)
        q4 = (xx >= 0) & (yy < 0)  # fourth quadrant (mirror of Q2)

        # create band-pass filters for each quadrant
        mask = t.zeros_like(radius, dtype=t.bool)
        mask |= q1 & ((radius <= self.r_high_q1) & (radius >= self.r_low_q1))
        mask |= q2 & ((radius <= self.r_high_q2) & (radius >= self.r_low_q2))
        mask |= q3 & ((radius <= self.r_high_q1) & (radius >= self.r_low_q1))
        mask |= q4 & ((radius <= self.r_high_q2) & (radius >= self.r_low_q2))

        # transform the img to the frequency domain
        fft_img = t.fft.fft2(img)
        fft_img = t.fft.fftshift(fft_img, dim=(-2, -1))

        # apply the filter mask
        fft_img_filtered = fft_img * mask[None, None, :, :]

        # transform back to the spatial domain
        fft_img_filtered = t.fft.ifftshift(fft_img_filtered, dim=(-2, -1))  # Shift back
        filtered_img = t.fft.ifft2(fft_img_filtered).real  # Take the real part
        filtered_img = filtered_img.view(*head_dim, C, H, W)

        return filtered_img


class RandomDFM(Transform):
    def __init__(self, subset):
        super().__init__()

        # create dfms
        input_pkl = f"./result/{subset}-dfm-data.pkl"
        with open(input_pkl, "rb") as pkl:
            dfm_params = load(pkl)["bin_dfm"]

        img_size = int(subset.split("-")[-1])
        self.dfm_list = []
        for key, value in dfm_params.items():
            r_low_q1, r_high_q1, r_low_q2, r_high_q2 = value
            self.dfm_list.append(createMask(img_size, img_size, r_low_q1, r_high_q1, r_low_q2, r_high_q2))

    def forward(self, img):
        if not isinstance(img, t.Tensor):
            raise TypeError("Input should be a torch Tensor.")

        # Ensure the input has at least 3 dimensions (C, H, W)
        if img.ndim < 3:
            raise ValueError("Input tensor must have at least 3 dimensions (C, H, W).")

        # handle batch dimensions
        head_dim = img.shape[:-3]
        C, H, W = img.shape[-3:]
        img = img.view(-1, C, H, W)

        # transform the img to the frequency domain
        fft_img = t.fft.fft2(img)
        fft_img = t.fft.fftshift(fft_img, dim=(-2, -1))

        # apply the filter mask
        mask = self.bin_dfm[random.choice(list(self.bin_dfm.keys()))]
        fft_img_filtered = fft_img * mask[None, None, :, :]

        # transform back to the spatial domain
        fft_img_filtered = t.fft.ifftshift(fft_img_filtered, dim=(-2, -1))  # Shift back
        filtered_img = t.fft.ifft2(fft_img_filtered).real  # Take the real part

        # Reshape back to original dimensions
        img = img.view(*head_dim, C, H, W)

        return img


class ZScoreNorm(Transform):
    def __init__(self):
        super().__init__()

    def forward(self, img):
        if not isinstance(img, t.Tensor):
            raise TypeError("Input should be a torch Tensor.")

        # Ensure the input has at least 3 dimensions (C, H, W)
        if img.ndim < 3:
            raise ValueError("Input tensor must have at least 3 dimensions (C, H, W).")

        # Handle batch dimensions
        head_dim = img.shape[:-3]
        C, H, W = img.shape[-3:]
        img = img.view(-1, C, H, W)  # Flatten leading dimensions into batch size

        # Compute mean and std along H and W dimensions for each channel
        mean = img.mean(dim=(-2, -1), keepdim=True)  # Shape: (N, C, 1, 1)
        std = img.std(dim=(-2, -1), keepdim=True)  # Shape: (N, C, 1, 1)

        # Normalize the img
        img = (img - mean) / (std + 1e-8)

        # Reshape back to original dimensions
        img = img.view(*head_dim, C, H, W)

        return img


class ButterworthLPF(Transform):
    def __init__(self, cutoff_ratio, order=2):

        super().__init__()
        if not (0 < cutoff_ratio <= 0.5):
            raise ValueError("cutoff_ratio must be in the range (0, 0.5].")
        self.cutoff_ratio = cutoff_ratio
        self.order = order

    def forward(self, img):

        if not isinstance(img, t.Tensor):
            raise TypeError("Input should be a torch Tensor.")

        if img.ndim < 3:
            raise ValueError("Input tensor must have at least 3 dimensions (C, H, W).")

        # Ensure the image is in float32 format for FFT processing
        img = img.float()

        # Extract dimensions
        *head_dims, channels, height, width = img.shape

        # Compute the cutoff frequency based on the ratio
        cutoff = self.cutoff_ratio * min(height, width)

        # Create the Butterworth filter in the frequency domain
        y = t.linspace(-height // 2, height // 2 - 1, height, device=img.device)
        x = t.linspace(-width // 2, width // 2 - 1, width, device=img.device)
        x_grid, y_grid = t.meshgrid(x, y, indexing="ij")
        distance = t.sqrt(x_grid ** 2 + y_grid ** 2)  # Distance from the center
        filter_mask = 1 / (1 + (distance / cutoff) ** (2 * self.order))

        # Apply FFT to the image
        fft_img = fft.fft2(img, dim=(-2, -1))
        fft_img_shifted = fft.fftshift(fft_img, dim=(-2, -1))

        # Apply the filter mask
        filtered_fft = fft_img_shifted * filter_mask

        # Perform the inverse FFT
        filtered_fft_shifted = fft.ifftshift(filtered_fft, dim=(-2, -1))
        filtered_img = fft.ifft2(filtered_fft_shifted, dim=(-2, -1))

        # Return the real part of the image, normalized to [0, 1]
        filtered_img_real = filtered_img.real
        filtered_img_real = t.clamp(filtered_img_real, 0.0, 1.0)  # Ensure the output remains within [0, 1]

        return filtered_img_real.view(*head_dims, channels, height, width)


def imgCrop(file_name, img):
    match = re.match(r"(\d{4})-([\w\-]+)-(pos|neg)-(\d+)_(\d+)_(\d+)_(\d+)\.png", file_name)
    if not match:
        raise ValueError("File name format is incorrect")

    img_idx, study, label, start_x, end_x, start_y, end_y = match.groups()
    start_y, end_y, start_x, end_x = map(int, [start_x, end_x, start_y, end_y])

    if img.ndim == 2:
        out_img = img[start_y:end_y, start_x:end_x]
    elif img.ndim == 3:
        out_img = img[start_y:end_y, start_x:end_x, :]
    else:
        raise ValueError("Unsupported img shape")

    return out_img


def applyBPF(img, mask):
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        mask = mask.unsqueeze(1)
    else:
        raise ValueError("Mask must have shape (H, W) or (B, H, W)")

    # transform to the frequency domain
    fft_img = t.fft.fft2(img)
    fft_img = t.fft.fftshift(fft_img, dim=(-2, -1))

    # apply filter
    fft_img_filtered = fft_img * mask

    # transform back to the spatial domain
    fft_img_filtered = t.fft.ifftshift(fft_img_filtered, dim=(-2, -1))  # Shift back
    filtered_img = t.fft.ifft2(fft_img_filtered).real  # Take the real part

    return filtered_img


class SpectrumEnergyLoss(nn.Module):
    def __init__(self):
        super(SpectrumEnergyLoss, self).__init__()

    def forward(self, img):
        """
        Compute spectrum energy loss for the given img batch.

        Args:
            img (t.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            t.Tensor: Scalar tensor representing the spectrum energy loss.
        """
        # Transform to frequency domain using FFT
        fft_img = t.fft.fft2(img, dim=(-2, -1))  # Apply FFT along height and width
        fft_magnitude = t.abs(fft_img)

        # Compute the total energy in the frequency domain
        spectrum_energy = t.sum(t.abs(fft_magnitude)) / img.numel()

        return spectrum_energy


class FreqAug(Transform):
    def __init__(self, fred_dict, exp_scale=7):
        super().__init__()
        self.freq_dict = fred_dict
        self.exp_scale = exp_scale

    def forward(self, img):
        if not isinstance(img, t.Tensor):
            raise TypeError("Input should be a torch Tensor.")

        # Ensure the input has at least 3 dimensions (C, H, W)
        if img.ndim < 3:
            raise ValueError("Input tensor must have at least 3 dimensions (C, H, W).")

        # Handle batch dimensions
        head_dim = img.shape[:-3]
        C, H, W = img.shape[-3:]
        img = img.view(-1, C, H, W)  # Flatten leading dimensions into batch size

        # randomly sample a phase, a frequency, and a blending scale
        # rand_key = random.choice(list(self.freq_dict.keys()))
        rand_key = 1
        rand_phase, rand_freq = random.choice(self.freq_dict[rand_key])
        rand_scale = abs(random.gauss(0, 0.5))

        # convert to tensor
        rand_phase = t.tensor(rand_phase)
        rand_freq = t.tensor(rand_freq)
        rand_scale = t.tensor(rand_scale)

        # make noise map
        yy = t.linspace(-H / 2, H / 2, steps=H)
        xx = t.linspace(-W / 2, W / 2, steps=W)
        ii, jj = t.meshgrid(yy, xx, indexing="ij")
        noise = t.sin(2 * t.pi * rand_freq * ii * t.cos(rand_phase) + jj * t.sin(rand_phase) - t.pi / 4)
        noise = (noise - t.min(noise)) / (t.max(noise) - t.min(noise))

        # blend
        out_img = (1 - rand_scale) * img + rand_scale * noise.view(1, 1, H, W)
        out_img = t.clamp(out_img - t.mean(out_img), 0, 1)

        # reshape back to original dimensions
        out_img = out_img.view(*head_dim, C, H, W)

        return out_img


def equalizeHist(img, contour):
    # create a binary mask from the contour
    mask = np.zeros_like(img, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)  # Fill the contour

    # extract pixels within the contour
    valid_pixels = img[mask > 0]

    # apply histogram equalization to the valid pixels
    hist = np.bincount(valid_pixels, minlength=65536)
    cdf = hist.cumsum()
    cdf_masked = np.ma.masked_equal(cdf, 0)
    cdf_min = cdf_masked.min()
    cdf_normalized = ((cdf_masked - cdf_min) * 65535 / (cdf_masked.max() - cdf_min)).filled(0).astype(np.uint16)

    # map the valid pixels back to the equalized values
    equalized_pixels = cdf_normalized[valid_pixels]

    # replace only the valid pixels
    equalized_image = img.copy()
    equalized_image[mask > 0] = equalized_pixels

    return equalized_image
