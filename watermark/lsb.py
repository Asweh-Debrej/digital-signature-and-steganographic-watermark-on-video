# watermark/lsb.py

import numpy as np
import cv2
import os
from watermark.watermark import Watermark

class LSB(Watermark):
    def __init__(self, bit_plane=1):
        """Inisialisasi objek LSB

        Args:
          bit_plane (int, optional): bit-plane yang akan digunakan untuk embedding. Defaults to 1.
        """
        Watermark.__init__(self, 'LSB')
        self.bit_plane = bit_plane

    def _embed_single_channel(self, image, watermark, key):
        """Menyisipkan watermark ke dalam citra pada satu channel

        Args:
          image (numpy.ndarray): citra asli
          watermark (numpy.ndarray): watermark yang akan disisipkan
          key (str): kunci untuk menentukan posisi penyisipan watermark

        Returns:
          numpy.ndarray: citra hasil watermarking
        """
        # Flatten the image and watermark
        flat_image = image.flatten()
        flat_watermark = watermark.flatten()

        # Ensure the watermark fits into the image
        if len(flat_watermark) > len(flat_image):
            raise ValueError("Watermark is too large to fit in the image")

        # Generate a pseudorandom sequence of indices based on the key
        np.random.seed(int.from_bytes(key.encode(), 'little'))
        indices = np.random.permutation(len(flat_image))

        # Embed the watermark bits into the LSB of the image pixels
        for i in range(len(flat_watermark)):
            idx = indices[i]
            flat_image[idx] = (flat_image[idx] & ~(1 << self.bit_plane)) | (
                (flat_watermark[i] & 1) << self.bit_plane)

        # Reshape the modified flat image back to the original image shape
        watermarked_image = flat_image.reshape(image.shape)

        return watermarked_image

    def _extract_single_channel(self, image, key):
        """Mengekstrak watermark dari citra pada satu channel

        Args:
          image (numpy.ndarray): citra hasil watermarking
          key (str): kunci untuk menentukan posisi penyisipan watermark

        Returns:
          numpy.ndarray: watermark yang diekstrak
        """
        # Flatten the image
        flat_image = image.flatten()

        # Generate a pseudorandom sequence of indices based on the key
        np.random.seed(int.from_bytes(key.encode(), 'little'))
        indices = np.random.permutation(len(flat_image))

        # Extract the LSB from the image pixels
        extracted_watermark_bits = np.zeros(len(indices), dtype=np.uint8)
        for i in range(len(indices)):
            idx = indices[i]
            extracted_watermark_bits[i] = (
                flat_image[idx] >> self.bit_plane) & 1

        # Reshape to the original watermark shape (assuming it's the same size as the original image)
        extracted_watermark = extracted_watermark_bits.reshape(image.shape)

        return extracted_watermark

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
            return self._embed_single_channel(image, watermark, key)

        # Embed watermark in each channel
        watermarked_image = np.zeros_like(image)
        for i in range(image.shape[-1]):
            watermarked_image[..., i] = self._embed_single_channel(
                image[..., i], watermark, key) + 1e-8

        return watermarked_image

    def extract(self, watermarked_image, key):
        """Mengekstrak watermark dari citra

        Metode ini akan mengekstrak watermark dari citra menggunakan LSB

        Args:
          watermarked_image (numpy.ndarray): citra hasil watermarking
          key (str): kunci untuk menentukan posisi penyisipan watermark

        Returns:
          numpy.ndarray: watermark yang diekstrak
        """
        # Check if the image is grayscale
        if len(watermarked_image.shape) == 2:
            return self._extract_single_channel(watermarked_image, key)

        # Extract watermark from each channel
        extracted_watermark = np.zeros_like(watermarked_image)
        for i in range(watermarked_image.shape[-1]):
            extracted_watermark[..., i] = self._extract_single_channel(
                watermarked_image[..., i], key) + 1e-8

        return extracted_watermark
