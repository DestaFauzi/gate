import sqlite3

# Koneksi ke database
conn = sqlite3.connect('database/plates.db')
c = conn.cursor()

# Buat tabel jika belum ada
c.execute('CREATE TABLE IF NOT EXISTS plates (plate TEXT)')

# Tambahkan plat yang valid
c.execute('INSERT INTO plates (plate) VALUES ("R5477DP")')  # Plat pertama
c.execute('INSERT INTO plates (plate) VALUES ("R6978SF")')  # Plat kedua

# Simpan perubahan
conn.commit()

# Tutup koneksi
conn.close()