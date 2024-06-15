# hash/sha2.py

import hashlib
from hash.hash import Hash

"""
Menggunakan SHA-256 untuk membuat hash data
"""
class SHA2(Hash):
  def __init__(self):
    """Inisialisasi objek SHA2
    """
    super().__init__('SHA-256')
    self.hash = hashlib.sha256

  def get_hash(self, data):
    """Menghasilkan hash dari data

    Args:
      data (str): data yang akan diambil hashnya

    Returns:
      str: hash dari data
    """
    return self.hash(data.encode()).hexdigest()

  def get_file_hash(self, file_path):
    """Menghasilkan hash dari file

    Args:
      file_path (str): path ke file yang akan diambil hashnya

    Returns:
      str: hash dari file
    """
    with open(file_path, 'rb') as file:
      return self.hash(file.read()).hexdigest()
