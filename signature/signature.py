from abc import ABC, abstractmethod

class Signature(ABC):
  def __init__(self, name):
    """Inisialisasi objek Signature

    Args:
      name (str): nama signature
    """
    self.name = name

    return self

  @abstractmethod
  def sign(self, data, private_key):
    """Menandatangani data

    Args:
      data (str): data yang akan ditandatangani
      private_key (str): kunci untuk menandatangani data

    Returns:
      str: tanda tangan digital dari data
    """
    pass

  @abstractmethod
  def verify(self, data, signature, public_key):
    """Memverifikasi tanda tangan digital

    Args:
      data (str): data yang akan diverifikasi
      signature (str): tanda tangan digital
      public_key (str): kunci publik untuk verifikasi

    Returns:
      bool: hasil verifikasi tanda tangan digital
    """
    pass
