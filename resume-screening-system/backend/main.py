from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from auth import verify_user, create_token, decode_token, create_user
from database import init_db, save_review, get_review, get_all_users
from ml_model import process_resume
import os

app = FastAPI()
init_db()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace "*" with frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/login")
def login(email: str = Form(...), password: str = Form(...), role: str = Form(None)):
    existing_role = verify_user(email, password)
    if existing_role:
        token = create_token({"email": email, "role": existing_role})
        return {"token": token, "role": existing_role}
    
    # New user flow
    if role not in ["applicant", "hr"]:
        raise HTTPException(status_code=401, detail="Invalid credentials or missing role for new user")
    
    create_user(email, password, role)
    token = create_token({"email": email, "role": role})
    return {"token": token, "role": role}

@app.post("/upload")
async def upload_cv(token: str = Form(...), file: UploadFile = File(...)):
    user = decode_token(token)
    contents = await file.read()
    os.makedirs("uploads", exist_ok=True)
    path = f"uploads/{file.filename}"
    with open(path, "wb") as f:
        f.write(contents)
    review = process_resume(path)
    save_review(user["email"], review)
    return {"review": review}

@app.get("/review")
def get_my_review(token: str):
    user = decode_token(token)
    return {"review": get_review(user["email"])}

@app.get("/applicants")
def list_applicants(token: str):
    user = decode_token(token)
    if user["role"] != "hr":
        raise HTTPException(status_code=403)
    return get_all_users()

@app.get("/applicant_review")
def applicant_review(email: str, token: str):
    user = decode_token(token)
    if user["role"] != "hr":
        raise HTTPException(status_code=403)
    return {"review": get_review(email)}
