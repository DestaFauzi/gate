from flask import Flask, render_template, jsonify
import sqlite3
import os

app = Flask(__name__)

# Buat folder dan database jika belum ada
if not os.path.exists('database'):
    os.makedirs('database')

conn = sqlite3.connect('database/plates.db')
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS plates (plate TEXT)')
conn.commit()
conn.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scan', methods=['GET'])
def scan():
    # Panggil fungsi pemindaian dari capture.py
    from capture import scan_plate
    scan_plate()  # Memanggil fungsi pemindaian
    return jsonify(status='Scanning started')

if __name__ == '__main__':
    app.run(debug=True)