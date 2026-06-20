import cv2
import urllib.request
import numpy as np
import face_recognition
import os
import time

# === KONFIGURASI ===
# URL ESP32-CAM (Sama seperti YOLO)
URL_KAMERA = 'http://10.17.148.118/capture'
FOLDER_DATASET = 'dataset'

def muat_wajah_dikenal(folder_dataset):
    print("Memuat wajah dari dataset... Mohon tunggu sebentar.")
    wajah_dikenal_encodings = []
    wajah_dikenal_names = []

    for nama_orang in os.listdir(folder_dataset):
        path_folder_orang = os.path.join(folder_dataset, nama_orang)
        
        if os.path.isdir(path_folder_orang):
            print(f"Memproses wajah untuk: {nama_orang}...")
            for nama_file in os.listdir(path_folder_orang):
                path_file = os.path.join(path_folder_orang, nama_file)
                
                try:
                    gambar = face_recognition.load_image_file(path_file)
                    encodings = face_recognition.face_encodings(gambar)
                    
                    if len(encodings) > 0:
                        wajah_dikenal_encodings.append(encodings[0])
                        wajah_dikenal_names.append(nama_orang)
                except Exception as e:
                    print(f"Gagal membaca {path_file}: {e}")

    print("Selesai memuat dataset!")
    return wajah_dikenal_encodings, wajah_dikenal_names

wajah_dikenal_encodings, wajah_dikenal_names = muat_wajah_dikenal(FOLDER_DATASET)

print("\nMemulai kamera ESP32-CAM...")
print("Tekan 'q' pada keyboard untuk berhenti.")

# Menggunakan fitur Stream (Video) dari ESP32-CAM agar jauh lebih cepat
# Port 81 adalah port khusus video stream bawaan dari ESP32-CAM
URL_STREAM = URL_KAMERA.replace("/capture", ":81/stream")
cap = cv2.VideoCapture(URL_STREAM)

# Atur buffer size sekecil mungkin agar tidak ada delay/gambar usang yang menumpuk
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

proses_frame_ini = True
lokasi_wajah = []
nama_wajah_terdeteksi = []

# 2. Looping kamera ESP32-CAM
while True:
    try:
        # Ambil frame dari Stream Video secara otomatis tanpa request HTTP berulang
        ret, frame = cap.read()
        if not ret:
            print("Video stream terputus!")
            break
        
        # Resize frame menjadi 1/4 ukuran agar proses lebih cepat (opsional, tapi sangat disarankan)
        frame_kecil = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        # Konversi warna gambar dari BGR (standar OpenCV) ke RGB (standar face_recognition)
        rgb_frame_kecil = np.ascontiguousarray(frame_kecil[:, :, ::-1])

        
        # Hanya proses wajah pada frame tertentu (misal: selang-seling) agar lebih enteng
        if proses_frame_ini:
            # Temukan semua wajah dan encoding wajah dalam frame kamera saat ini
            # model="hog" digunakan agar lebih ringan untuk CPU
            lokasi_wajah = face_recognition.face_locations(rgb_frame_kecil, model="hog")
            encodings_wajah = face_recognition.face_encodings(rgb_frame_kecil, lokasi_wajah)

            nama_wajah_terdeteksi = []

            # Coba cocokkan setiap wajah yang ada di kamera
            for encoding_wajah in encodings_wajah:
                # Bandingkan dengan database kita
                cocok = face_recognition.compare_faces(wajah_dikenal_encodings, encoding_wajah, tolerance=0.5)
                nama = "Tidak Dikenal"

                # Jika ada kemiripan, cari yang paling mirip
                jarak_wajah = face_recognition.face_distance(wajah_dikenal_encodings, encoding_wajah)
                if len(jarak_wajah) > 0:
                    index_paling_mirip = np.argmin(jarak_wajah)
                    if cocok[index_paling_mirip]:
                        nama = wajah_dikenal_names[index_paling_mirip]

                nama_wajah_terdeteksi.append(nama)

        # Ubah status proses frame untuk frame berikutnya (jika True jadi False, jika False jadi True)
        proses_frame_ini = not proses_frame_ini

        # Gambar kotak di layar
        for (atas, kanan, bawah, kiri), nama in zip(lokasi_wajah, nama_wajah_terdeteksi):
            # Kembalikan ke ukuran asli karena tadi kita kecilkan 1/4
            atas *= 4
            kanan *= 4
            bawah *= 4
            kiri *= 4

            # Gambar kotak di wajah
            warna_kotak = (0, 255, 0) if nama != "Tidak Dikenal" else (0, 0, 255)
            cv2.rectangle(frame, (kiri, atas), (kanan, bawah), warna_kotak, 2)

            # Gambar label nama
            cv2.rectangle(frame, (kiri, bawah - 35), (kanan, bawah), warna_kotak, cv2.FILLED)
            cv2.putText(frame, nama, (kiri + 6, bawah - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

        # Tampilkan hasilnya
        cv2.imshow('Sistem Pengenalan Wajah', frame)

        # Keluar jika tekan 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as e:
        print("Error/Gagal mengambil frame:", e)
        break

cv2.destroyAllWindows()
