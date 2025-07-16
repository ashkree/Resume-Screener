from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from auth import verify_user, create_token, decode_token, create_user
from database import init_db, save_review, get_review, get_all_users
from ml_model import process_resume
import os

# Initialize FastAPI app
app = FastAPI()

# Initialize the database connection and tables (we used NeonDB for our system)
init_db()

# Allow frontend (Netlify) to access backend (Render) via CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://irsas.netlify.app/"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Login endpoint: Handles both existing (cross-checked with out database) and first-time users
@app.post("/login")
def login(email: str = Form(...), password: str = Form(...), role: str = Form(None)):
    # Check if user exists in db
    existing_role = verify_user(email, password)
    if existing_role:
        # Return token and role if valid credentials are entered
        token = create_token({"email": email, "role": existing_role})
        return {"token": token, "role": existing_role}
    
    # If not existing, create a new user in db only if valid role is selected
    if role not in ["applicant", "hr"]:
        raise HTTPException(status_code=401, detail="Invalid credentials or missing role for new user")
    
    # Register the new user and return their token
    create_user(email, password, role)
    token = create_token({"email": email, "role": role})
    return {"token": token, "role": role}

# Upload CV endpoint (Applicant role only): Accepts and saves file, then generates review
@app.post("/upload")
async def upload_cv(token: str = Form(...), file: UploadFile = File(...)):
    user = decode_token(token)  # Verify user 
    contents = await file.read()

    # Save uploaded file locally
    os.makedirs("uploads", exist_ok=True)
    path = f"uploads/{file.filename}"
    with open(path, "wb") as f:
        f.write(contents)

    # Process CV using our ML model
    review = process_resume(path)

    # Save review result and filename in database
    save_review(user["email"], review, file.filename)
    return {"review": review}

# Endpoint for applicants to retrieve their own review
@app.get("/review")
def get_my_review(token: str):
    user = decode_token(token)
    result = get_review(user["email"])
    return {"review": result["review"], "filename": result["filename"]}

# HR-specific endpoint to list all users who have submitted resumes to our system
@app.get("/applicants")
def list_applicants(token: str):
    user = decode_token(token)
    if user["role"] != "hr":
        raise HTTPException(status_code=403)  # Error raised when non-HR users try to access
    return get_all_users()

# HR-only endpoint to get review of a specific applicant
@app.get("/applicant_review")
def applicant_review(email: str, token: str):
    user = decode_token(token)
    if user["role"] != "hr":
        raise HTTPException(status_code=403)
    result = get_review(email)
    return {"review": result["review"], "filename": result["filename"]}
