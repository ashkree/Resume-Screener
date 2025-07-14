import sqlite3

def init_db():
    conn = sqlite3.connect("data.db")
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS reviews (email TEXT PRIMARY KEY, review TEXT)")
    conn.commit()

def save_review(email, review):
    conn = sqlite3.connect("data.db")
    c = conn.cursor()
    c.execute("REPLACE INTO reviews VALUES (?, ?)", (email, review))
    conn.commit()

def get_review(email):
    conn = sqlite3.connect("data.db")
    c = conn.cursor()
    c.execute("SELECT review FROM reviews WHERE email=?", (email,))
    row = c.fetchone()
    return row[0] if row else "No review yet."

def get_all_users():
    conn = sqlite3.connect("data.db")
    c = conn.cursor()
    c.execute("SELECT email FROM reviews")
    return [row[0] for row in c.fetchall()]
