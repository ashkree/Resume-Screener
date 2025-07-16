from jose import jwt, JWTError
from passlib.hash import bcrypt
from database import get_user, add_user

SECRET = "secret123"

def verify_user(email, password):
    user = get_user(email)
    if user and bcrypt.verify(password, user[1]):  # user[1] = hashed password
        return user[2]  # user[2] = role
    return None

def create_user(email, password, role):
    hashed_password = bcrypt.hash(password)
    add_user(email, hashed_password, role)

def create_token(data: dict):
    return jwt.encode(data, SECRET, algorithm="HS256")

def decode_token(token: str):
    try:
        return jwt.decode(token, SECRET, algorithms=["HS256"])
    except JWTError:
        raise
