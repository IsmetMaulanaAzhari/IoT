import requests
from datetime import datetime
import time

# === UBAH BAGIAN INI ===
# Masukkan IP Address ESP32-CAM Anda
ip_address = "172.20.100.4" 
url = f"http://{ip_address}/capture"
# =======================

def jepret_foto():
    print(f"Mencoba mengambil foto dari {url}...")
    try:
        # Meminta foto dari ESP32-CAM
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            # Membuat nama file unik berdasarkan waktu saat ini
            waktu = datetime.now().strftime("%Y%m%d_%H%M%S")
            nama_file = f"hasil_foto_{waktu}.jpg"
            
            # Menyimpan foto ke laptop
            with open(nama_file, "wb") as f:
                f.write(response.content)
            print(f"SUKSES! Foto disimpan dengan nama: {nama_file}")
        else:
            print(f"Gagal memotret. Status code: {response.status_code}")
            
    except Exception as e:
        print(f"Error: Tidak dapat terhubung ke ESP32-CAM. {e}")

# Jalankan fungsinya
jepret_foto()