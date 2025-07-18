from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from auth import verify_user, create_token, decode_token, create_user
from database import init_db, save_review, get_review, get_all_users, get_user
from ml_model import parse_resume, process_resume
import os

# Initialize FastAPI app
app = FastAPI()

# Initialize the database connection and tables (we used NeonDB for our system)
init_db()

# Allow frontend (Netlify) to access backend (Render) via CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Login endpoint: Handles both existing (cross-checked with our database) and first-time users
@app.post("/login")
def login(email: str = Form(...), password: str = Form(...), role: str = Form(None)):
    existing_user = get_user(email)
    if existing_user:
        if verify_user(email, password):
            token = create_token({"email": email, "role": existing_user[2]})
            return {"token": token, "role": existing_user[2]}
        else:
            raise HTTPException(status_code=401, detail="Incorrect password")
    if role not in ["applicant", "hr"]:
        raise HTTPException(status_code=401, detail="Invalid credentials or missing role for new user")
    create_user(email, password, role)
    token = create_token({"email": email, "role": role})
    return {"token": token, "role": role}

# Upload CV endpoint (Applicant role only): Accepts and saves file, parses and reviews it
@app.post("/upload")
async def upload_cv(token: str = Form(...), file: UploadFile = File(...)):
    try:
        user = decode_token(token)
    except:
        raise HTTPException(status_code=401, detail="Invalid token")

    contents = await file.read()
    os.makedirs("uploads", exist_ok=True)
    path = f"uploads/{file.filename}"
    with open(path, "wb") as f:
        f.write(contents)

    # Step 1: Extract raw text from resume
    parsed_text = parse_resume(path)

    # Step 2: Feed parsed text into ML model to get review
    review = process_resume(parsed_text)

    # Step 3: Store filename + parsed text + review
    save_review(user["email"], parsed_text, review, file.filename)

    return {"review": review}

# Endpoint for applicants to retrieve their own review
@app.get("/review")
def get_my_review(token: str):
    try:
        user = decode_token(token)
    except:
        raise HTTPException(status_code=401, detail="Invalid token")

    result = get_review(user["email"])
    return {"review": result["review"], "filename": result["filename"]}

# HR-specific endpoint to list all users who have submitted resumes to our system
@app.get("/applicants")
def list_applicants(token: str):
    try:
        user = decode_token(token)
    except:
        raise HTTPException(status_code=401, detail="Invalid token")

    if user["role"] != "hr":
        raise HTTPException(status_code=403)
    return get_all_users()

# HR-only endpoint to get review of a specific applicant
@app.get("/applicant_review")
def applicant_review(email: str, token: str):
    try:
        user = decode_token(token)
    except:
        raise HTTPException(status_code=401, detail="Invalid token")

    if user["role"] != "hr":
        raise HTTPException(status_code=403)
    result = get_review(email)
    return {"review": result["review"], "filename": result["filename"]}
