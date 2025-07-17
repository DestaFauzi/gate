import cv2
import sqlite3
import pytesseract

# Konfigurasi Tesseract (pastikan path sesuai)
pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

def scan_plate():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Kamera tidak dapat diakses.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Gagal mengambil gambar.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

        plate_text = pytesseract.image_to_string(thresh, config='--psm 8').strip()

        if plate_text:
            conn = sqlite3.connect('database/plates.db')
            c = conn.cursor()
            c.execute('SELECT * FROM plates WHERE plate = ?', (plate_text,))
            if c.fetchone():
                print(f"Plat terdeteksi: {plate_text} - Validasi berhasil!")
            else:
                print(f"Plat terdeteksi: {plate_text} - Plat tidak ditemukan.")
            conn.close()

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()