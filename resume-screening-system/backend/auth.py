from jose import jwt, JWTError
from passlib.hash import bcrypt

SECRET = "secret123"
USERS = {
    "applicant@example.com": {"password": bcrypt.hash("1234"), "role": "applicant"},
    "hr@example.com": {"password": bcrypt.hash("1234"), "role": "hr"}
}

def verify_user(email, password):
    user = USERS.get(email)
    if user and bcrypt.verify(password, user["password"]):
        return user["role"]
    return None

def create_token(data: dict):
    return jwt.encode(data, SECRET, algorithm="HS256")

def decode_token(token: str):
    try:
        return jwt.decode(token, SECRET, algorithms=["HS256"])
    except JWTError:
        raise
