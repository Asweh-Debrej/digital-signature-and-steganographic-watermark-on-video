# /signature.py

"""Digital signature untuk video
Program ini akan menghasilkan digital signature dari video yang diberikan
"""

import cv2
import numpy as np
import os
import sys
import hashlib
import time
import json


def get_signature(video_path):
    """Menghasilkan digital signature dari video

    Args:
        video_path (str): path ke video yang akan diambil digital signaturenya

    Returns:
        str: digital signature dari video
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_interval = int(fps / 2)

    signature = []
    for i in range(0, frame_count, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (8, 8))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        signature.append(frame)

    cap.release()
    return hashlib.md5(np.array(signature).tobytes()).hexdigest()


def main():
    if len(sys.argv) != 2:
        print("Usage: python signature.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print("File not found")
        sys.exit(1)

    print("Generating signature...")

    signature = get_signature(video_path)
    print(signature)

    with open("signature.json", "w") as f:
        json.dump({"signature": signature}, f)

    print("Signature saved to signature.json")


if __name__ == "__main__":
    main()
