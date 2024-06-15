# watermark/dwt.py

import numpy as np
import pywt
import cv2
from watermark.watermark import Watermark
from scipy.ndimage import gaussian_filter

"""Kelas untuk melakukan Discrete Wavelet Transform
Kelas memiliki beberapa metode untuk melakukan DWT pada citra
Watermark disisipkan dalam sub-band tertentu menggunakan kunci untuk menentukan posisi yang tepat
"""


class DWT(Watermark):
    def __init__(self, wavelet='haar', mode='symmetric', k = 0.03):
        """Inisialisasi objek DWT

        Args:
          wavelet (str, optional): nama wavelet. Defaults to 'haar'.
          mode (str, optional): mode. Defaults to 'symmetric'.
        """
        Watermark.__init__(self, 'DWT')
        self.wavelet = wavelet
        self.mode = mode
        self.k = k # alpha

    def _embed_single_channel(self, image, watermark):
        """Menyisipkan watermark ke dalam citra pada satu channel

        Args:
          image (numpy.ndarray): citra asli
          watermark (numpy.ndarray): watermark yang akan disisipkan
          key (str): kunci untuk menentukan posisi penyisipan watermark

        Returns:
          numpy.ndarray: citra hasil watermarking
        """
        # Apply DWT to the image
        coeffs_image = pywt.dwt2(image, self.wavelet, mode=self.mode)
        LL_image, (LH_image, HL_image, HH_image) = coeffs_image

        coeffs_watermark = pywt.dwt2(watermark, self.wavelet, mode=self.mode)
        LL_watermark, (LH_watermark, HL_watermark, HH_watermark) = coeffs_watermark

        # Embed the watermark in the DWT coefficients
        LL_watermarked = LL_image + self.k * LL_watermark
        LH_watermarked = LH_image + self.k * LH_watermark
        HL_watermarked = HL_image + self.k * HL_watermark
        HH_watermarked = HH_image + self.k * HH_watermark

        coeffs_watermarked = LL_watermarked, (LH_watermarked, HL_watermarked, HH_watermarked)

        watermarked_image = pywt.idwt2(coeffs_watermarked, self.wavelet, mode=self.mode)

        return watermarked_image

    def _extract_single_channel(self, watermarked, original):
        """Mengekstrak watermark dari citra pada satu channel

        Args:
          image (numpy.ndarray): citra hasil watermarking

        Returns:
          numpy.ndarray: watermark yang diekstrak
        """
        # Apply DWT to the watermarked image
        coeffs_watermarked = pywt.dwt2(watermarked, self.wavelet, mode=self.mode)
        LL_watermarked, (LH_watermarked, HL_watermarked, HH_watermarked) = coeffs_watermarked

        coeffs_original = pywt.dwt2(original, self.wavelet, mode=self.mode)
        LL_original, (LH_original, HL_original, HH_original) = coeffs_original

        # Extract the watermark from the DWT coefficients
        LL_extracted = (LL_watermarked - LL_original) / self.k
        LH_extracted = (LH_watermarked - LH_original) / self.k
        HL_extracted = (HL_watermarked - HL_original) / self.k
        HH_extracted = (HH_watermarked - HH_original) / self.k

        coeffs_extracted = LL_extracted, (LH_extracted, HL_extracted, HH_extracted)

        extracted_watermark = pywt.idwt2(coeffs_extracted, self.wavelet, mode=self.mode)

        return extracted_watermark

    def embed(self, image, watermark, key):
        """Menyisipkan watermark ke dalam citra
        Metode ini akan menyisipkan watermark ke dalam citra menggunakan DWT

        Args:
          image (numpy.ndarray): citra asli
          watermark (numpy.ndarray): watermark yang akan disisipkan
          key (str): kunci untuk menentukan posisi penyisipan watermark

        Returns:
          numpy.ndarray: citra hasil watermarking
        """
        # Check if the image is grayscale
        if len(image.shape) == 2:
            return self._embed_single_channel(image, watermark)

        # Embed watermark in each channel
        watermarked_image = np.zeros_like(image)
        for i in range(image.shape[-1]):
            watermarked_image[..., i] = self._embed_single_channel(image[..., i], watermark[..., i])

        return watermarked_image

    def extract(self, watermarked, original, key):
        """Mengekstrak watermark dari citra
        Metode ini akan mengekstrak watermark dari citra menggunakan DWT

        Args:
          watermarked_image (numpy.ndarray): citra hasil watermarking
          key (str): kunci untuk menentukan posisi penyisipan watermark

        Returns:
          numpy.ndarray: watermark hasil ekstraksi
        """
        # Check if the image is grayscale
        if len(watermarked.shape) == 2:
            return self._extract_single_channel(watermarked, original)

        # Extract watermark from each channel
        extracted_watermark = np.zeros_like(watermarked)
        for i in range(watermarked.shape[-1]):
            extracted_watermark[..., i] = self._extract_single_channel(watermarked[..., i], original[..., i])

        return extracted_watermark

