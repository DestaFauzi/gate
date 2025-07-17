import cv2
import numpy as np
import pytesseract
import time
# from threading import Thread, Lock # Tidak diperlukan tanpa motor stepper
# import RPi.GPIO as GPIO # Tidak diperlukan tanpa motor stepper
from datetime import datetime
import re
import json

# --- Konfigurasi Sistem ---
class Config:
    """Konfigurasi sistem"""
    
    # Konfigurasi kamera
    CAMERA_INDEX = 0 # Biasanya 0 untuk webcam bawaan atau kamera USB pertama
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    
    # Konfigurasi OCR (Tesseract)
    # Pastikan Tesseract sudah terinstal dan path-nya benar
    TESSERACT_PATH = '/usr/bin/tesseract'
    
    # Bahasa Tesseract. Gunakan 'eng' (default) atau 'ind' jika tersedia.
    # UNTUK AKURASI TERBAIK, setelah melatih Tesseract secara kustom,
    # ganti ini dengan nama bahasa kustom Anda (misal: 'ind_lp')
    TESSERACT_LANG = 'eng' 
    
    # --oem 3: Menggunakan engine Tesseract terbaru (LSTM)
    # --psm 8: Page Segmentation Mode untuk single word/line of text (cocok untuk plat)
    # -c tessedit_char_whitelist: Membatasi karakter yang dikenali hanya pada huruf kapital dan angka
    OCR_CONFIG = f'-l {TESSERACT_LANG} --oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    
    # Konfigurasi deteksi plat nomor (berbasis kontur)
    MIN_PLATE_AREA = 1000 # Luas area minimum untuk dianggap sebagai kandidat plat
    MIN_PLATE_WIDTH = 100 # Lebar minimum kandidat plat
    MIN_PLATE_HEIGHT = 25 # Tinggi minimum kandidat plat
    MIN_ASPECT_RATIO = 2.0 # Rasio aspek (lebar/tinggi) minimum plat
    MAX_ASPECT_RATIO = 4.5 # Rasio aspek maksimum plat
    
    # Konfigurasi sistem kontrol (disimpan tapi hanya sebagai placeholder untuk cooldown deteksi OCR)
    DETECTION_COOLDOWN = 3  # Detik: Waktu tunggu setelah deteksi valid sebelum deteksi berikutnya diproses
    MAX_PLATE_CANDIDATES = 3 # Jumlah kandidat plat teratas yang akan diproses OCR
    
    # File log
    LOG_FILE = 'detection_log.json' # Mengubah nama file log agar terpisah
    
    # Daftar plat nomor yang diizinkan (DIUBAH MENJADI SET UNTUK PENCARIAN LEBIH CEPAT)
    AUTHORIZED_PLATES = {"R5477DP", "R6978SF"} # Menggunakan set literal {}

# --- Kelas Logger ---
class Logger:
    def __init__(self):
        self.log_file = Config.LOG_FILE
    
    def log_detection(self, plate_text, status):
        """Log deteksi ke file JSON."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'plate_number': plate_text,
            'status': status # 'authorized' atau 'unauthorized'
        }
        
        try:
            with open(self.log_file, 'a+') as f:
                f.seek(0)
                content = f.read().strip()
                
                data = []
                if content:
                    try:
                        if content.startswith('[') and content.endswith(']'):
                            data = json.loads(content)
                        else:
                            data = [json.loads(line) for line in content.split('\n') if line.strip()]
                    except json.JSONDecodeError:
                        print(f"Warning: Log file '{self.log_file}' contains invalid JSON. Starting fresh.")
                        data = []
                
                data.append(log_entry)
                
                f.seek(0)
                f.truncate()
                json.dump(data, f, indent=4)
                
        except Exception as e:
            print(f"Error logging: {e}")

# --- Kelas Plate Detector ---
class PlateDetector:
    def __init__(self):
        self.plate_patterns = [
            re.compile(r'^[A-Z]{1,2}\d{1,4}[A-Z]{1,3}$'), 
            re.compile(r'^[A-Z]{1,2}\s?\d{1,4}\s?[A-Z]{1,3}$'), 
            re.compile(r'^[A-Z]\d{4}[A-Z]{2}$')
        ]
    
    def preprocess_for_detection(self, image):
        """Preprocessing gambar untuk deteksi area plat nomor."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        return edges
    
    def detect_plate_areas(self, edges):
        """Deteksi area yang mungkin merupakan plat nomor berdasarkan kontur."""
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        plate_candidates = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > Config.MIN_PLATE_AREA:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                if (Config.MIN_ASPECT_RATIO <= aspect_ratio <= Config.MAX_ASPECT_RATIO 
                    and w > Config.MIN_PLATE_WIDTH and h > Config.MIN_PLATE_HEIGHT):
                    plate_candidates.append((x, y, w, h))
        
        plate_candidates.sort(key=lambda rect: rect[2]*rect[3], reverse=True)
        return plate_candidates[:Config.MAX_PLATE_CANDIDATES]
    
    def preprocess_for_ocr(self, plate_roi):
        """Preprocessing khusus untuk gambar ROI plat nomor sebelum dikirim ke OCR."""
        gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
        
        avg_intensity = np.mean(gray)
        
        if avg_intensity < 128: 
            threshold_type = cv2.THRESH_BINARY_INV
            # print("Detected dark plate (e.g., black plate with white text), using THRESH_BINARY_INV") # Debugging
        else: 
            threshold_type = cv2.THRESH_BINARY
            # print("Detected light plate (e.g., white plate with black text), using THRESH_BINARY") # Debugging

        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        threshold_type, 11, 2)
        
        denoised = cv2.medianBlur(thresh, 3) 

        kernel = np.ones((2,2), np.uint8)
        processed_img = cv2.dilate(denoised, kernel, iterations=1) 

        resized = cv2.resize(processed_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        return resized
    
    def post_process_ocr_result(self, text):
        """Melakukan post-processing pada hasil OCR untuk memperbaiki kesalahan umum."""
        clean_text = text.strip().upper().replace(' ', '').replace('\n', '')

        clean_text = clean_text.replace('O', '0')
        clean_text = clean_text.replace('I', '1')
        clean_text = clean_text.replace('L', '1')
        clean_text = clean_text.replace('S', '5')
        clean_text = clean_text.replace('Z', '2')
        clean_text = clean_text.replace('B', '8')
        clean_text = clean_text.replace('G', '6')
        clean_text = clean_text.replace('Q', '0')

        clean_text = re.sub(r'[^A-Z0-9]', '', clean_text)
        
        return clean_text

    def extract_plate_text(self, image, bbox):
        """Ekstrak teks dari area plat yang terdeteksi menggunakan Tesseract OCR."""
        x, y, w, h = bbox
        y_max = min(y + h, image.shape[0])
        x_max = min(x + w, image.shape[1])
        plate_roi = image[y:y_max, x:x_max]
        
        if plate_roi.shape[0] == 0 or plate_roi.shape[1] == 0:
            return ""
        
        ocr_ready = self.preprocess_for_ocr(plate_roi)
        
        try:
            raw_text = pytesseract.image_to_string(ocr_ready, config=Config.OCR_CONFIG)
            final_text = self.post_process_ocr_result(raw_text)
            
            print(f"Raw OCR: '{raw_text.strip()}' => Processed: '{final_text}'")
            return final_text
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""
    
    def validate_plate_format(self, plate_text):
        """Validasi format plat nomor menggunakan regex yang sudah dikompilasi."""
        if not plate_text or len(plate_text) < 4:
            return False
        
        for pattern in self.plate_patterns:
            if pattern.match(plate_text):
                return True
        return False

# --- Kelas Sistem Pengenalan Plat Nomor Utama (Hanya Deteksi) ---
class PlateRecognitionSystem:
    def __init__(self):
        self.plate_detector = PlateDetector()
        self.logger = Logger()
        self.cap = cv2.VideoCapture(Config.CAMERA_INDEX)
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
        
        if not self.cap.isOpened():
            raise IOError("Tidak dapat membuka kamera. Pastikan kamera terhubung dan indeks benar.")

        self.last_detection_time = 0 # Waktu terakhir plat nomor valid terdeteksi
        
        # Set path Tesseract
        pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_PATH
        
    def process_frame(self, frame):
        """Proses frame untuk deteksi plat nomor"""
        edges = self.plate_detector.preprocess_for_detection(frame)
        plate_areas = self.plate_detector.detect_plate_areas(edges)
        
        detected_plates = []
        
        for area in plate_areas:
            x, y, w, h = area
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            plate_text = self.plate_detector.extract_plate_text(frame, area)
            
            if plate_text and self.plate_detector.validate_plate_format(plate_text):
                detected_plates.append({
                    'text': plate_text,
                    'bbox': (x, y, w, h)
                })
                cv2.putText(frame, plate_text, (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return frame, detected_plates

    def handle_plate_detection(self, plate_text):
        """
        Menangani logika setelah plat nomor terdeteksi dan divalidasi.
        Fokus pada logging dan tampilan status.
        """
        current_time = time.time()
        
        # Cek cooldown untuk mencegah deteksi berulang terlalu cepat
        if current_time - self.last_detection_time < Config.DETECTION_COOLDOWN:
            return # Masih dalam masa cooldown, abaikan deteksi ini
        
        is_authorized = plate_text in Config.AUTHORIZED_PLATES
        
        if is_authorized:
            print(f"Plat '{plate_text}' terdeteksi. STATUS: DISETUJUI.")
            self.logger.log_detection(plate_text, 'authorized')
        else:
            print(f"Plat '{plate_text}' terdeteksi. STATUS: TIDAK DISETUJUI.")
            self.logger.log_detection(plate_text, 'unauthorized')
            
        self.last_detection_time = current_time # Update waktu deteksi terakhir

    def run(self):
        """Jalankan sistem deteksi plat nomor utama."""
        print("=== Sistem Deteksi Plat Nomor ===")
        print("Daftar plat yang diizinkan:")
        for plate in Config.AUTHORIZED_PLATES:
            print(f"- {plate}")
        
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Gagal membaca frame kamera. Pastikan kamera berfungsi.")
                    break
                
                processed_frame, detected_plates = self.process_frame(frame)
                
                for plate_info in detected_plates:
                    self.handle_plate_detection(plate_info['text'])
                
                # Tampilkan status deteksi pada frame
                status_display_text = "Mendeteksi Plat Nomor..."
                cv2.putText(processed_frame, status_display_text, (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA) # Warna kuning
                
                cv2.imshow('Plate Detection', processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            print(f"Terjadi kesalahan fatal: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Bersihkan sumber daya sebelum keluar."""
        print("Membersihkan sumber daya...")
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        print("Sistem dimatikan.")

# --- Main Execution ---
if __name__ == "__main__":
    prs = PlateRecognitionSystem()
    prs.run()