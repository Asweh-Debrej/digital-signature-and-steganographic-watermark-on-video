# signature/sha2.py

"""
Menggunakan RSA-2048 untuk enkripsi digital signature
"""

from Cryptodome.PublicKey import RSA
from Cryptodome.Cipher import PKCS1_OAEP
from signature.signature import Signature

class RSA2048(Signature):
  def __init__(self):
    """Inisialisasi objek RSA2048
    """
    super().__init__('RSA-2048')
    self.key = RSA.generate(2048)
    self.public_key = self.key.export_key()
    self.private_key = self.key.publickey().export_key()

  def sign(self, data, private_key):
    """Mengenkripsi data dengan RSA-2048

    Args:
      data (str): data yang akan dienkripsi
      private_key (str): kunci privat enkripsi, tidak akan digunakan

    Returns:
      bytes: data yang telah dienkripsi
    """
    cipher_rsa = PKCS1_OAEP.new(self.key)
    return cipher_rsa.encrypt(data.encode())

  def verify(self, data):
    """Mendekripsi data dengan RSA-2048

    Args:
      data (bytes): data yang akan didekripsi

    Returns:
      str: data yang telah didekripsi
    """
    cipher_rsa = PKCS1_OAEP.new(self.key)
    return cipher_rsa.decrypt(data).decode()
