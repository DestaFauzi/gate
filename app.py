from flask import Flask, render_template, Response
import sqlite3
import RPi.GPIO as GPIO
import cv2
import pytesseract
import os

app = Flask(__name__)

# Setup GPIO untuk motor
GPIO.setmode(GPIO.BCM)
motor_pins = [17, 18, 27, 22]  # Ganti dengan pin yang digunakan
for pin in motor_pins:
    GPIO.setup(pin, GPIO.OUT)

# Fungsi untuk membuka gerbang
def open_gate():
    for _ in range(100):  # Ubah sesuai kebutuhan
        for pin in motor_pins:
            GPIO.output(pin, GPIO.HIGH)
            GPIO.output(pin, GPIO.LOW)

# Mengalirkan video
def generate_frames():
    camera = cv2.VideoCapture(0)  # Ganti dengan index kamera jika perlu
    while True:
        success, frame = camera.read()  # Baca frame dari kamera
        if not success:
            break
        else:
            # Mengubah frame menjadi JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/scan')
def scan():
    plate = detect_plate()
    if plate:
        if plate_found_in_db(plate):
            open_gate()
            return f"Plate {plate} matched! Gate opened."
        return "Plate not found."
    return "No plate detected."

def capture_image():
    camera = cv2.VideoCapture(0)
    ret, frame = camera.read()
    if ret:
        cv2.imwrite('plate.jpg', frame)
    camera.release()

def detect_plate():
    image = cv2.imread('plate.jpg')
    plate_text = pytesseract.image_to_string(image, config='--psm 8')
    return plate_text.strip()

def plate_found_in_db(plate):
    conn = sqlite3.connect('database/plates.db')
    c = conn.cursor()
    c.execute('SELECT * FROM plates WHERE plate = ?', (plate,))
    result = c.fetchone()
    conn.close()
    return result is not None

if __name__ == '__main__':
    # Buat folder dan database jika belum ada
    if not os.path.exists('database'):
        os.makedirs('database')

    conn = sqlite3.connect('database/plates.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS plates (plate TEXT)')
    conn.commit()
    conn.close()

    app.run(host='0.0.0.0', port=5000)