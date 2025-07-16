import psycopg2
import os

# Get the database connection URL from the environment (NeonDB)
DATABASE_URL = os.getenv("DATABASE_URL")

# Establish a connection to the database
def get_conn():
    return psycopg2.connect(DATABASE_URL)

# Initialize tables if they don't already exist
def init_db():
    conn = get_conn()
    cur = conn.cursor()

    # Users table: stores login credentials and roles
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            role TEXT NOT NULL
        )
    """)

    # Reviews table: stores resume review + filename, linked to a registered user
    cur.execute("""
        CREATE TABLE IF NOT EXISTS reviews (
            email TEXT PRIMARY KEY,
            review TEXT,
            filename TEXT,
            FOREIGN KEY (email) REFERENCES users(email)
        )
    """)

    conn.commit()
    cur.close()
    conn.close()

# Fetch a user by their email
def get_user(email):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT email, password, role FROM users WHERE email=%s", (email,))
    user = cur.fetchone()
    cur.close()
    conn.close()
    return user  # returns (email, password, role) or None

# Add a new user only if they don't already exist
def add_user(email, hashed_password, role):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO users (email, password, role)
        VALUES (%s, %s, %s)
        ON CONFLICT DO NOTHING
    """, (email, hashed_password, role))
    conn.commit()
    cur.close()
    conn.close()

# Save or update a review and the uploaded filename
def save_review(email, review, filename):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO reviews (email, review, filename)
        VALUES (%s, %s, %s)
        ON CONFLICT (email)
        DO UPDATE SET review = EXCLUDED.review, filename = EXCLUDED.filename
    """, (email, review, filename))
    conn.commit()
    cur.close()
    conn.close()

# Retrieve both the review and filename for a given user
def get_review(email):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT review, filename FROM reviews WHERE email=%s", (email,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row:
        return {"review": row[0], "filename": row[1]}
    else:
        return {"review": "No review yet.", "filename": "No file uploaded."}

# Get a list of all user emails (used by HR dashboard)
def get_all_users():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("SELECT email FROM users")
    emails = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()
    return emails
