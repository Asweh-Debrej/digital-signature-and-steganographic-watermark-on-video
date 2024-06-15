from abc import ABC, abstractmethod

class Watermark:
  def __init__(self, name):
    """Inisialisasi objek Watermark

    Args:
      name (str): nama watermark
    """
    self.name = name

    return self

  @abstractmethod
  def embed(self, image, watermark, key):
    """Menyisipkan watermark ke dalam citra

    Args:
      image (numpy.ndarray): citra asli
      watermark (numpy.ndarray): watermark yang akan disisipkan
      key (str): kunci untuk menentukan posisi penyisipan watermark

    Returns:
      numpy.ndarray: citra hasil watermarking
    """
    pass

  @abstractmethod
  def extract(self, watermarked, original, key):
    """Mengekstrak watermark dari citra

    Args:
      watermarked (numpy.ndarray): citra hasil watermarking
      original (numpy.ndarray): citra asli
      key (str): kunci untuk menentukan posisi penyisipan watermark

    Returns:
      numpy.ndarray: watermark hasil ekstraksi
    """
    pass
