# watermark/dsss.py

import numpy as np
import cv2
import os

from watermark.watermark import Watermark
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve


class DSSS(Watermark):
    def __init__(self, k=10):
        """Inisialisasi objek DSSS

        Args:
          k (float, optional): scaling factor untuk watermark. Defaults to 0.005.
        """
        Watermark.__init__(self, 'DSSS')
        self.k = k # Penguat watermark

    def _enhance(self, image):
        """Meningkatkan citra menggunakan gaussian filter

        Args:
          image (numpy.ndarray): citra yang akan ditingkatkan

        Returns:
          numpy.ndarray: citra yang telah ditingkatkan
        """
        # Membuat kernel filter
        kernel = np.array([[-1, -1, -1],
                          [-1,  8, -1],
                          [-1, -1, -1]])

        # Menerapkan filter
        filtered = cv2.filter2D(image.astype(np.int16), -1, kernel)
        return filtered

    def generate_pseudo_random_sequence(self, shape, key):
        """Menghasilkan urutan pseudorandom

        Args:
          shape (tuple): bentuk urutan yang dihasilkan
          key (str): kunci untuk menentukan urutan pseudorandom

        Returns:
          numpy.ndarray: urutan pseudorandom
        """
        np.random.seed(int.from_bytes(key.encode(), 'little'))
        pseudo_random_sequence = np.random.choice([1, -1], size=shape)
        return pseudo_random_sequence.astype(np.int16)

    def _detect_single_channel(self, watermarked, key):
        """Mendeteksi watermark pada citra pada satu channel

        Args:
          watermarked (numpy.ndarray): citra hasil watermarking
          key (str): kunci untuk menentukan urutan pseudorandom

        Returns:
          int: nilai korelasi antara watermark yang diekstrak dan watermark asli
        """
        # Generate pseudorandom sequence
        pseudo_random_sequence = self.generate_pseudo_random_sequence(
            watermarked.shape, key)

        # Enhance using gaussian filter
        watermarked = self._enhance(watermarked)

        result = cv2.matchTemplate(watermarked.astype(np.float32), pseudo_random_sequence.astype(np.float32), cv2.TM_CCORR_NORMED)

        _, _, _, max_loc = cv2.minMaxLoc(result)

        correlation = result[max_loc[1], max_loc[0]]

        return correlation

    def _embed_single_channel(self, image, key):
        """Menyisipkan watermark ke dalam citra pada satu channel

        Args:
          image (numpy.ndarray): citra asli
          watermark (numpy.ndarray): watermark yang akan disisipkan

        Returns:
          numpy.ndarray: citra hasil watermarking
        """

        # Generate pseudorandom sequence
        pseudo_random_sequence = self.generate_pseudo_random_sequence(
            image.shape, key)

        # Embed the watermark using DSSS
        watermarked_image = cv2.add(image.astype(np.int16), self.k * pseudo_random_sequence)

        # Convert below 0 to 0 and above 255 to 255
        watermarked_image = np.clip(watermarked_image, 0, 255).astype(np.uint8)

        return watermarked_image


    def embed(self, image, watermark, key):
        """Menyisipkan watermark ke dalam citra

        Metode ini akan menyisipkan watermark ke dalam citra menggunakan LSB

        Args:
          image (numpy.ndarray): citra asli
          watermark (numpy.ndarray): watermark yang akan disisipkan
          key (str): kunci untuk menentukan posisi penyisipan watermark

        Returns:
          numpy.ndarray: citra hasil watermarking
        """
        # Check if the image is grayscale
        if len(image.shape) == 2:
            return self._embed_single_channel(image, key)

        # Embed watermark in each channel
        watermarked_image = np.zeros_like(image)
        for i in range(image.shape[-1]):
            watermarked_image[..., i] = self._embed_single_channel(
                image[..., i], key)

        return watermarked_image

    def detect(self, watermarked, key):
        """Mendeteksi watermark pada citra

        Metode ini akan mendeteksi watermark pada citra menggunakan DSSS

        Args:
          watermarked (numpy.ndarray): citra hasil watermarking
          key (str): kunci untuk menentukan urutan pseudorandom

        Returns:
          int: nilai korelasi antara watermark yang diekstrak dan watermark asli
        """
        # Check if the image is grayscale
        if len(watermarked.shape) == 2:
            return self._detect_single_channel(watermarked, key)

        # Detect watermark in each channel
        correlations = []
        for i in range(watermarked.shape[-1]):
            correlations.append(self._detect_single_channel(
                watermarked[..., i], key))

        return correlations
