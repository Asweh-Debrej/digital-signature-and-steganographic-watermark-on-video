# watermark/dct.py

import numpy as np
import cv2
from watermark.watermark import Watermark
from scipy.ndimage import gaussian_filter
from scipy.fftpack import dct, idct


class DCT(Watermark):
    def __init__(self, k=0.006):
        """Inisialisasi objek DCT

        Args:
          k (int, optional): scaling factor untuk watermark. Defaults to 2.
        """
        Watermark.__init__(self, 'DCT')
        self.k = k

    def _embed_single_channel(self, image, watermark):
        """Menyisipkan watermark ke dalam citra pada satu channel

        Args:
          image (numpy.ndarray): citra asli
          watermark (numpy.ndarray): watermark yang akan disisipkan

        Returns:
          numpy.ndarray: citra hasil watermarking
        """
        # Apply DCT to the image
        image_dct = dct(dct(image.T, norm='ortho').T, norm='ortho')
        watermark_dct = dct(dct(watermark.T, norm='ortho').T, norm='ortho')

        # Embed the watermark in the DCT coefficients
        image_dct += self.k * watermark_dct

        watermarked_image = idct(
            idct(image_dct.T, norm='ortho').T, norm='ortho')

        return watermarked_image

    def _extract_single_channel(self, watermarked, original):
        """Mengekstrak watermark dari citra pada satu channel

        Args:
          watermarked (numpy.ndarray): citra hasil watermarking
          original (numpy.ndarray): citra asli

        Returns:
          numpy.ndarray: watermark yang diekstrak
        """
        # Apply DCT to the images
        watermarked_dct = dct(dct(watermarked.T, norm='ortho').T, norm='ortho')
        original_dct = dct(dct(original.T, norm='ortho').T, norm='ortho')

        # Extract the watermark from the DCT coefficients
        watermark_dct = (watermarked_dct - original_dct) / self.k

        watermark = idct(idct(watermark_dct.T, norm='ortho').T, norm='ortho')

        return watermark

    def embed(self, image, watermark, key=None):
        """Menyisipkan watermark ke dalam citra

        Args:
          image (numpy.ndarray): citra asli
          watermark (numpy.ndarray): watermark yang akan disisipkan

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

    def extract(self, watermarked, original, key=None):
        """Mengekstrak watermark dari citra

        Args:
          watermarked (numpy.ndarray): citra hasil watermarking
          original (numpy.ndarray): citra asli

        Returns:
          numpy.ndarray: watermark yang diekstrak
        """
        # Check if the image is grayscale
        if len(watermarked.shape) == 2:
            return self._extract_single_channel(watermarked, original)

        # Extract watermark from each channel
        watermark = np.zeros_like(watermarked)
        for i in range(watermarked.shape[-1]):
            watermark[..., i] = self._extract_single_channel(watermarked[..., i], original[..., i])

        return watermark
