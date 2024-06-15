from abc import ABC, abstractmethod

class Hash(ABC):
  def __init__(self, name):
    """Inisialisasi objek Hash

    Args:
      name (str): nama hash
    """
    self.name = name

    return self

  @abstractmethod
  def get_hash(self, data):
    """Menghasilkan hash dari data

    Args:
      data (str): data yang akan diambil hashnya

    Returns:
      str: hash dari data
    """
    pass

  @abstractmethod
  def get_file_hash(self, file_path):
    """Menghasilkan hash dari file

    Args:
      file_path (str): path ke file yang akan diambil hashnya

    Returns:
      str: hash dari file
    """
    pass
