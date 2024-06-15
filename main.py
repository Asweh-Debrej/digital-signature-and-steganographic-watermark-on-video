# main.py

"""Program utama
Program utama untuk melakukan serangkaian proses pada video.

Program menjalankan fungsi-fungsi yang terdefinisi dari beberapa algoritma watermarking dan digital signature.
Urutan proses utama yang dilakukan adalah:
1. Menerapkan watermarking pada video, watermark secara opsional dapat dibuat oleh algoritma terpilih
2. Menghasilkan digital signature dari video yang telah di-watermark
3. Mengenkripsi digital signature dengan kunci tertentu
4. Menyimpan hasil watermarking dan digital signature ke dalam file terpisah
5. Membaca hasil watermarking dan digital signature dari file
6. Mendekripsi digital signature
7. Memverifikasi digital signature dengan video yang telah di-watermark
8. Memberikan evaluasi hasil verifikasi
9. Menampilkan hasil evaluasi
"""

from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import time
import numpy as np
import cv2
import json
import os
from watermark.watermark import Watermark
from watermark.dwt import DWT
from watermark.dsss import DSSS
from watermark.dct import DCT
from signature.signature import Signature
from signature.rsa import RSA2048
from hash.hash import Hash
from hash.sha2 import SHA2
from hash.sha3 import SHA3

watermarks = [
    DSSS(),
    DWT(),
    DCT()
]

signatures = [
    RSA2048(),
]

hashes = [
    SHA2(),
    SHA3()
]


FRAME_COUNT = 100


def calculate_psnr(original, watermarked):
    """Menghitung nilai PSNR antara dua citra

    Args:
      original (numpy.ndarray): citra asli
      watermarked (numpy.ndarray): citra hasil watermarking

    Returns:
      float: nilai PSNR
    """
    return psnr(original, watermarked)


def calculate_ssim(original, watermarked):
    """Menghitung nilai SSIM antara dua citra

    Args:
      original (numpy.ndarray): citra asli
      watermarked (numpy.ndarray): citra hasil watermarking

    Returns:
      float: nilai SSIM
    """
    multichannel = len(original.shape) == 3
    channel_axis = 2 if multichannel else None
    return ssim(original, watermarked, multichannel=multichannel, channel_axis=channel_axis, data_range=watermarked.max() - watermarked.min())


def calculate_nc(original_watermark, extracted_watermark):
    """Menghitung nilai NC antara dua watermark

    Args:
      original_watermark (numpy.ndarray): watermark asli
      extracted_watermark (numpy.ndarray): watermark hasil ekstraksi

    Returns:
      float: nilai NC
    """
    return np.corrcoef(original_watermark.flatten(), extracted_watermark.flatten())[0, 1]


def calculate_ber(original_watermark, extracted_watermark):
    """Menghitung nilai BER antara dua watermark

    Args:
      original_watermark (numpy.ndarray): watermark asli
      extracted_watermark (numpy.ndarray): watermark hasil ekstraksi

    Returns:
      float: nilai BER
    """
    # menggunakan numpy unpackbits untuk mendapatkan nilai bit dari watermark
    original_bits = np.unpackbits(original_watermark.flatten())
    extracted_bits = np.unpackbits(extracted_watermark.flatten())

    # menghitung jumlah bit yang berbeda
    return np.sum(original_bits != extracted_bits) / len(original_bits)


def calculate_capacity(video_path):
    """Menghitung kapasitas watermarking pada video

    Args:
      video_path (str): path ke video yang akan dihitung kapasitasnya

    Returns:
      int: kapasitas watermarking
    """
    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return frame_count * width * height


def apply_attack(video_path, attack_type, output_path):
    """Menerapkan serangan pada video

    Args:
      video_path (str): path ke video yang akan diserang
      attack_type (str): tipe serangan yang akan diterapkan
      output_path (str): path ke video hasil serangan

    Returns:
      None
    """
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for _ in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break

        if attack_type == 'noise':
            noise = np.random.normal(0, 10, frame.shape).astype(np.uint8)
            frame = cv2.add(frame, noise, dtype=cv2.CV_8UC3)
        elif attack_type == 'brightness':
            frame = cv2.convertScaleAbs(frame, alpha=1.5, beta=0)
        elif attack_type == 'contrast':
            frame = cv2.convertScaleAbs(frame, alpha=1, beta=50)
        elif attack_type == 'rotation':
            rows, cols, _ = frame.shape
            M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 1, 1)
            frame = cv2.warpAffine(frame, M, (cols, rows))
        elif attack_type == 'compression':
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

        out.write(frame)

    cap.release()
    out.release()


def evaluate_watermarking(watermarking: Watermark, frames, watermark, watermarking_key, frame_count=FRAME_COUNT, width=0, height=0, fps=0, folder=''):
    """Mengevaluasi hasil watermarking pada video

    Args:
      Watermark_class (object): kelas algoritma watermarking
      video_path (str): path ke video yang akan di-watermark
      watermark_path (str): path ke watermark yang akan di-embed
      watermarking_key (str): kunci watermarking

    Returns:
      dict: hasil evaluasi dengan format {'psnr': float, 'ssim': float, 'nc': float, 'ber': float, 'capacity': int, 'time': float, 'attack': {{'psnr': float, 'ssim': float, 'nc': float, 'ber': float}}
    """
    print(f'Evaluating {watermarking.name}')

    # output frames dan watermark original
    original_frame_path = f'{folder}/original_frames.mp4'
    os.makedirs(os.path.dirname(original_frame_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(original_frame_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

    original_watermark_path = f'{folder}/original_watermark.png'
    os.makedirs(os.path.dirname(original_watermark_path), exist_ok=True)
    cv2.imwrite(original_watermark_path, watermark)

    # Menerapkan watermarking
    watermarked_frames = []
    start_time = time.time()

    print('Embedding watermark')
    print('_'*FRAME_COUNT)
    for i in range(FRAME_COUNT):
        watermarked_frame = watermarking.embed(
            frames[i], watermark, watermarking_key)
        watermarked_frames.append(watermarked_frame)
        print('█', end='', flush=True)

    print('\tDone!')
    print('‾'*FRAME_COUNT)

    # Menghitung waktu watermarking
    watermarking_time = time.time() - start_time

    # Membuat video hasil watermarking
    watermarking_path = f'{folder}/watermarked_video.mp4'
    os.makedirs(os.path.dirname(watermarking_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(watermarking_path, fourcc, fps, (width, height))
    for frame in watermarked_frames:
        out.write(frame)
    out.release()

    # Menghitung kapasitas watermarking
    # capacity = watermarking.calculate_capacity(video_path)

    # ekstrak watermark
    extracted_watermarks = []

    print('Extracting watermark')
    print('_'*FRAME_COUNT)
    for i in range(FRAME_COUNT):
        extracted_watermark = watermarking.extract(
            watermarked_frames[i], frames[i], watermarking_key)
        extracted_watermarks.append(extracted_watermark)
        print('█', end='', flush=True)

    print('\tDone!')
    print('‾'*FRAME_COUNT)

    extracted_watermark_path = f'{folder}/extracted_watermark.png'
    os.makedirs(os.path.dirname(extracted_watermark_path), exist_ok=True)
    cv2.imwrite(extracted_watermark_path, extracted_watermarks[0])

    # Menghitung PSNR dan SSIM
    psnr_values = []
    ssim_values = []

    print('Calculating PSNR and SSIM')
    print('_'*FRAME_COUNT)
    for i in range(FRAME_COUNT):
        psnr_value = calculate_psnr(frames[i], watermarked_frames[i])
        ssim_value = calculate_ssim(frames[i], watermarked_frames[i])
        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
        print('█', end='', flush=True)

    print('\tDone!')
    print('‾'*FRAME_COUNT)

    nc_values = []
    ber_values = []

    print('Calculating NC and BER')
    print('_'*FRAME_COUNT)
    for i in range(FRAME_COUNT):
        nc_value = calculate_nc(watermark, extracted_watermarks[i])
        ber_value = calculate_ber(watermark, extracted_watermarks[i])
        nc_values.append(nc_value)
        ber_values.append(ber_value)
        print('█', end='', flush=True)

    print('\tDone!')
    print('‾'*FRAME_COUNT)

    # Menerapkan serangan pada video hasil watermarking
    print('Applying attacks...')
    attacks = [
        'noise',
        'brightness',
        'contrast',
        'rotation',
        'compression'
    ]
    attack_results = {}
    for attack in attacks:
        print(f'Applying {attack} attack...')
        attacked_video_path = f'{folder}/{attack}_attack.mp4'
        os.makedirs(os.path.dirname(attacked_video_path), exist_ok=True)
        apply_attack(watermarking_path, attack, attacked_video_path)

        # Membaca video hasil serangan
        attacked_video = cv2.VideoCapture(attacked_video_path)
        attacked_frames = []

        print('Reading attacked video')
        print('_'*FRAME_COUNT)
        for _ in range(FRAME_COUNT):
            ret, frame = attacked_video.read()
            if not ret:
                break

            attacked_frames.append(frame)
            print('█', end='', flush=True)

        print('\tDone!')
        print('‾'*FRAME_COUNT)

        # Menghitung PSNR dan SSIM
        attacked_psnr_values = []
        attacked_ssim_values = []

        print('Calculating PSNR and SSIM')
        print('_'*FRAME_COUNT)
        for i in range(FRAME_COUNT):
            psnr_value = calculate_psnr(
                attacked_frames[i], watermarked_frames[i])
            ssim_value = calculate_ssim(
                attacked_frames[i], watermarked_frames[i])
            attacked_psnr_values.append(psnr_value)
            attacked_ssim_values.append(ssim_value)
            print('█', end='', flush=True)

        print('\tDone!')
        print('‾'*FRAME_COUNT)

        # Menghitung NC dan BER
        extracted_watermarks = []

        print('Extracting watermark')
        print('_'*FRAME_COUNT)
        for i in range(FRAME_COUNT):
            extracted_watermark = watermarking.extract(
                attacked_frames[i], frames[i], watermarking_key)
            extracted_watermarks.append(extracted_watermark)
            print('█', end='', flush=True)

        print('\tDone!')
        print('‾'*FRAME_COUNT)

        extracted_attacked_watermark_path = f'{folder}/{attack}_extracted_watermark.png'
        os.makedirs(os.path.dirname(extracted_attacked_watermark_path), exist_ok=True)
        cv2.imwrite(extracted_attacked_watermark_path, extracted_watermarks[0])

        attacked_nc_values = []
        attacked_ber_values = []

        print('Calculating NC and BER')
        print('_'*FRAME_COUNT)
        for i in range(FRAME_COUNT):
            nc_value = calculate_nc(watermark, extracted_watermarks[i])
            ber_value = calculate_ber(watermark, extracted_watermarks[i])
            attacked_nc_values.append(nc_value)
            attacked_ber_values.append(ber_value)
            print('█', end='', flush=True)

        print('\tDone!')
        print('‾'*FRAME_COUNT)

        attack_results[attack] = {
            'psnr': sum(attacked_psnr_values) / len(attacked_psnr_values),
            'ssim': sum(attacked_ssim_values) / len(attacked_ssim_values),
            'nc': sum(attacked_nc_values) / len(attacked_nc_values),
            'ber': sum(attacked_ber_values) / len(attacked_ber_values),
        }

        print(f'{attack} done')

    # Menyimpan hasil watermarking
    results = {
        'name': f'{watermarking.name}',
        'time': watermarking_time,
        'time per frame': watermarking_time / FRAME_COUNT,
        'psnr': sum(psnr_values) / len(psnr_values),
        'ssim': sum(ssim_values) / len(ssim_values),
        'nc': sum(nc_values) / len(nc_values),
        'ber': sum(ber_values) / len(ber_values),
        # 'capacity': capacity,
        'attack': attack_results,
    }

    return results


def evaluate_dsss(frames, watermarking: DSSS, watermarking_key, frame_count=FRAME_COUNT, width=0, height=0, fps=0, folder=''):
    """Mengevaluasi hasil watermarking pada video, khusus untuk DSSS

    Args:
        frames (list): list of frames
        watermark (numpy.ndarray): watermark
        watermarking_key (str): kunci watermarking
        frame_count (int): jumlah frame yang akan di-watermark
        width (int): lebar frame
        height (int): tinggi frame
        fps (int): frame per second
        folder (str): folder untuk menyimpan hasil evaluasi

    Returns:
        dict: hasil evaluasi
    """
    print(f'Evaluating DSSS')

    # output frames dan watermark original
    original_frame_path = f'{folder}/original_frames.mp4'
    os.makedirs(os.path.dirname(original_frame_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(original_frame_path, fourcc, fps, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

    watermark = np.zeros_like(frames[0])
    for i in range(frames[0].shape[-1]):
        watermark[..., i] = watermarking.generate_pseudo_random_sequence(
            (height, width), watermarking_key)

    watermark_path = f'{folder}/watermark.png'
    os.makedirs(os.path.dirname(watermark_path), exist_ok=True)
    cv2.imwrite(watermark_path, watermark)

    # Menerapkan watermarking
    watermarked_frames = []
    start_time = time.time()

    print('Embedding watermark')
    print('_'*FRAME_COUNT)
    for i in range(FRAME_COUNT):
        watermarked_frame = watermarking.embed(
            frames[i], watermark, watermarking_key)
        watermarked_frames.append(watermarked_frame)
        print('█', end='', flush=True)

    print('\tDone!')
    print('‾'*FRAME_COUNT)

    # Menghitung waktu watermarking
    watermarking_time = time.time() - start_time

    # Membuat video hasil watermarking
    watermarking_path = f'{folder}/watermarked_video.mp4'
    os.makedirs(os.path.dirname(watermarking_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(watermarking_path, fourcc, fps, (width, height))
    for frame in watermarked_frames:
        out.write(frame)
    out.release()

    # Menghitung kapasitas watermarking
    # capacity = watermarking.calculate_capacity(video_path)

    # Ambil nilai korelasi watermark dengan video
    print('Detecting watermark')
    print('_'*FRAME_COUNT)
    correlations = []
    for i in range(FRAME_COUNT):
        correlation = watermarking.detect(
            watermarked_frames[i], watermarking_key)
        correlation_value = sum(correlation) / len(correlation)
        correlations.append(correlation_value)
        print('█', end='', flush=True)

    print('\tDone!')
    print('‾'*FRAME_COUNT)

    # Menerapkan serangan pada video hasil watermarking
    print('Applying attacks...')
    attacks = [
        'noise',
        'brightness',
        'contrast',
        'rotation',
        'compression'
    ]
    attack_results = {}
    for attack in attacks:
        print(f'Applying {attack} attack...')
        attacked_video_path = f'{folder}/{attack}_attack.mp4'
        os.makedirs(os.path.dirname(attacked_video_path), exist_ok=True)
        apply_attack(watermarking_path, attack, attacked_video_path)

        # Membaca video hasil serangan
        attacked_video = cv2.VideoCapture(attacked_video_path)
        attacked_frames = []

        print('Reading attacked video')
        print('_'*FRAME_COUNT)
        for _ in range(FRAME_COUNT):
            ret, frame = attacked_video.read()
            if not ret:
                break

            attacked_frames.append(frame)
            print('█', end='', flush=True)

        print('\tDone!')
        print('‾'*FRAME_COUNT)

        # Menghitung korelasi
        attacked_correlations = []
        for i in range(FRAME_COUNT):
            correlation = watermarking.detect(
                attacked_frames[i], watermarking_key)
            correlation_value = sum(correlation) / len(correlation)
            attacked_correlations.append(correlation_value)

        attack_results[attack] = {
            'correlation': sum(attacked_correlations) / len(attacked_correlations),
        }

        print(f'{attack} done')

    # Menyimpan hasil watermarking
    results = {
        'name': f'DSSS',
        'time': watermarking_time,
        'time per frame': watermarking_time / FRAME_COUNT,
        'correlation': sum(correlations) / len(correlations),
        'attack': attack_results,
    }

    return results


def sign(video_path, signature: Signature, hash: Hash, key):
    """Membuat digital signature dari video

    Args:
      video_path (str): path ke video yang akan di-sign
      signature (object): kelas algoritma signature
      hash (object): kelas algoritma hash
      key (str): kunci untuk menandatangani video

    Returns:
      str: digital signature
    """
    video = cv2.VideoCapture(video_path)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    start_time = time.time()

    video_hash = hash.get_file_hash(video_path)
    signature_value = signature.sign(video_hash, key)

    signature_time = time.time() - start_time

    return (signature_value, video_hash, signature_time)


def generate_random_watermark(shape):
    """Membuat watermark acak dengan ukuran tertentu

    Args:
      shape (tuple): ukuran watermark

    Returns:
      numpy.ndarray: watermark acak dengan nilai -255 atau 255
    """
    watermark = np.random.randint(0, 2, shape).astype(np.uint8)
    return watermark * 255


def main():
    video_path = 'video.mp4'
    watermark_path = 'watermark.png'
    watermarking_key = 'key'
    key = 'key'

    # import grayscale
    video = cv2.VideoCapture(video_path)

    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    video_shape = (height, width, 3)

    print('Reading, there are', frame_count, 'frames. Only',
          FRAME_COUNT, 'frames will be processed')
    print('_'*FRAME_COUNT)
    frames = []
    for i in range(FRAME_COUNT):
        ret, frame = video.read()
        if not ret:
            break

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frames.append(frame)
        print('█', end='', flush=True)
    print('\tDone!')
    print('‾'*FRAME_COUNT)
    # # Membuat watermark acak sebagai image
    # watermark = generate_random_watermark(video_shape)

    # import watermark 3 channel
    watermark = cv2.imread(watermark_path, cv2.IMREAD_COLOR)
    watermark = cv2.resize(watermark, (width, height))

    results = []
    for wm in watermarks:
        if wm.name == 'DSSS':
            result = evaluate_dsss(frames, wm, watermarking_key,
                                   frame_count=FRAME_COUNT, width=width, height=height, fps=fps, folder=f'./{wm.name}')
        else:
            result = evaluate_watermarking(wm, frames, watermark, watermarking_key,
                                           frame_count=FRAME_COUNT, width=width, height=height, fps=fps, folder=f'./{wm.name}')

        result['signature'] = []
        for signature in signatures:
            for hash in hashes:
                signature_value, hash_value, signature_time = sign(
                    f'./{wm.name}/watermarked_video.mp4', signature, hash, key)

                result['signature'].append({
                    'name': f'{signature.name} - {hash.name}',
                    'value': str(signature_value),
                    'hash': hash_value,
                    'time': signature_time,
                })

                print(f'{wm.name} - {signature.name} - {hash.name} done')

        results.append(result)

        with open(f'./{wm.name}/result.json', 'w') as f:
            json.dump(result, f, indent=2)

    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)



if __name__ == '__main__':
    main()
