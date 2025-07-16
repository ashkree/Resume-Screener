import psycopg2
import os

DATABASE_URL = os.getenv("DATABASE_URL")

def get_conn():
    return psycopg2.connect(DATABASE_URL)

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            role TEXT NOT NULL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            email TEXT PRIMARY KEY,
            review TEXT,
            FOREIGN KEY (email) REFERENCES users(email)
        )
    """)
    conn.commit()
    cur.close()
    conn.close()

def get_user(email):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT email, password, role FROM users WHERE email=%s", (email,))
    user = cur.fetchone()
    cur.close()
    conn.close()
    return user  # (email, hashed_password, role) or None

def add_user(email, hashed_password, role):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("INSERT INTO users (email, password, role) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                (email, hashed_password, role))
    conn.commit()
    cur.close()
    conn.close()

def save_review(email, review):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO reviews (email, review) VALUES (%s, %s)
        ON CONFLICT (email) DO UPDATE SET review = EXCLUDED.review
    """, (email, review))
    conn.commit()
    cur.close()
    conn.close()

def get_review(email):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT review FROM reviews WHERE email=%s", (email,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row[0] if row else "No review yet."

def get_all_users():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT email FROM users")
    emails = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()
    return emails
