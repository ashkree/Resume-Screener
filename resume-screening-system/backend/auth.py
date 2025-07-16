from jose import jwt, JWTError  
from passlib.hash import bcrypt  
from database import get_user, add_user  

# Secret key used to sign JWT tokens 
# JWT (JSON Web Token) is a secure way to represent user identity and send it between systems, often for authentication.
SECRET = "secret123"

# Checks whether a user's email/password matches what's in our db
def verify_user(email, password):
    user = get_user(email)  # Fetching user tuple (email, hashed_pw, role)
    if user and bcrypt.verify(password, user[1]):  # Compare hashed password
        return user[2]  # Return role (either 'applicant' or 'hr') if valid
    return None  # If no match, return None

# Creates a new user by hashing their input password and saving to db
def create_user(email, password, role):
    hashed_password = bcrypt.hash(password)  
    add_user(email, hashed_password, role)  

# Generates a JWT token based on entered user data
def create_token(data: dict):
    return jwt.encode(data, SECRET, algorithm="HS256")

# Decodes a token and verifies it using the secret we made at the beginning
def decode_token(token: str):
    try:
        return jwt.decode(token, SECRET, algorithms=["HS256"])
    except JWTError:
        raise  # Raise error if token is invalid or expired
