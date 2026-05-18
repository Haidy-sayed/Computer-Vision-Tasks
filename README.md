# FilterLab
**Image Processing · Computer Vision · Python · PyQt5**

A desktop application for interactive image processing, implementing histogram analysis and equalization, spatial domain filtering, and frequency domain filtering — built to demonstrate core computer vision concepts through real-time visual feedback.

---

## Overview

FilterLab allows users to load grayscale images and apply a pipeline of classical image processing operations, observing the effect of each step immediately. The application covers three fundamental areas of digital image processing: intensity distribution analysis via histograms, spatial convolution filters, and Fourier-based frequency domain filters.

---

## Key Features

- **Histogram Analysis & Equalization** — computes and displays the intensity histogram of a loaded image, applies histogram equalization using a cumulative distribution function mapping, and renders the equalized image alongside its corrected histogram
- **Spatial Domain Filters** — median blur for noise reduction and Laplacian edge detection, with the frequency spectrum of the filtered result displayed alongside
- **Frequency Domain Filters** — FFT-based low-pass (centre-preserving mask) and high-pass (centre-zeroing mask) filters with live magnitude spectrum visualization
- **Export** — any displayed canvas can be saved as PNG or JPEG

---

## Technical Implementation

### Histogram Equalization
Equalization is implemented from scratch using the CDF mapping formula:

$$M(i) = \frac{CDF(i)}{H \times W} \times L - 1$$

where $L = 256$ grey levels and $H \times W$ is the image resolution.

### Frequency Domain Filtering
Images are transformed using `numpy.fft.fft2` and shifted with `fftshift`. Filters are applied as masks in the frequency domain before inverse transformation:

```python
# Low pass — keep centre
mask[crow-50:crow+51, ccol-50:ccol+51] = 1
filtered = mask * fshift

# High pass — zero centre
fshift[crow-30:crow+31, ccol-30:ccol+31] = 0
```

The magnitude spectrum is normalised to uint8 before display to avoid float overflow in PIL.

---

## Stack

`Python 3` · `PyQt5` · `OpenCV` · `NumPy` · `Matplotlib` · `Pillow`

```bash
pip install PyQt5 opencv-python numpy matplotlib Pillow
python Task1CV_fixed.py
```

---

## Skills Demonstrated
Histogram equalization · Fourier transform filtering · Spatial convolution · NumPy image processing · PyQt5 GUI development · Computer vision fundamentals
