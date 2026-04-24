import cv2
import urllib.request
import numpy as np
from ultralytics import YOLO

# 1. Memanggil model AI YOLOv8 (Otomatis download model teringan 'yolov8n.pt' saat pertama kali dijalankan)
print("Memuat otak YOLO...")
model = YOLO('yolov8n.pt') 

# === UBAH IP ADDRESS JIKA BERUBAH ===
url = 'http://172.20.100.4/capture'
# ====================================

print("Memulai aliran kamera... (Tekan 'q' pada keyboard untuk berhenti)")

# Looping agar kamera terus mengambil gambar seperti video
while True:
    try:
        # 2. Mengambil gambar dari ESP32-CAM
        img_resp = urllib.request.urlopen(url)
        imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
        frame = cv2.imdecode(imgnp, -1)

        # 3. Proses gambar menggunakan YOLO!
        # YOLO akan mencari objek di dalam 'frame'
        hasil = model(frame, verbose=False) 

        # 4. Menggambar kotak (bounding box) di atas foto hasil deteksi
        frame_hasil = hasil[0].plot()

        # 5. Menampilkan foto ke layar laptop
        cv2.imshow('Deteksi Objek ESP32-CAM + YOLO', frame_hasil)

        # Jika tombol 'q' ditekan, program berhenti
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print("Gagal mengambil gambar dari kamera:", e)
        break

cv2.destroyAllWindows()