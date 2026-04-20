# Smart Attendance System (Streamlit + Face Recognition)

Production-ready portfolio project for placements: a smart face-recognition attendance web app with modern Streamlit UI.

## Features

- Beautiful UI with title: `🚀 Smart Attendance System`
- Sidebar student registration from multiple uploaded photos
- Encodings persisted in `encodings.pkl`
- Main screen camera snapshot via `st.camera_input`
- Fallback image upload when camera is unavailable
- Face detection + recognition with green bounding boxes and labels
- Attendance auto-marked in `Attendance.csv` (`Name`, `Time`, `Date`)
- One attendance entry per student per day
- Beginner-friendly comments for interview explanation

## Project Structure

```text
smart-attendance-system/
├── app.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── render.yaml
├── Attendance.csv
├── student_images/
├── ploomber-cloud.yaml
└── encodings.pkl           # auto-created after registration
```

## Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Render (Recommended)

Repository: `govindsharma2001/smart-attendance-system`

### 1) Push latest code to GitHub

```bash
git add .
git commit -m "Add Docker + Render deployment setup"
git push origin main
```

### 2) Deploy from Render dashboard (manual method)

1. Open [https://dashboard.render.com/](https://dashboard.render.com/)
2. Click **New +** -> **Web Service**
3. Connect GitHub and select repo: `govindsharma2001/smart-attendance-system`
4. Render will detect `Dockerfile` automatically
5. Set:
   - **Name**: `smart-attendance-system` (or any name)
   - **Runtime**: Docker
   - **Branch**: `main`
   - **Plan**: Free
6. Click **Create Web Service**

Render will build and deploy using:
- `Dockerfile` (installs system + Python dependencies)
- start command already inside Dockerfile:
  `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT`

### 3) Deploy using Blueprint (optional)

Since `render.yaml` is included, you can deploy from **Blueprint**:
1. In Render: **New +** -> **Blueprint**
2. Select the same repo
3. Render reads `render.yaml` and creates the service

### 4) Verify deployment

- Open the Render app URL
- Register students from sidebar
- Capture/upload image and validate recognition
- Check green boxes + labels on detected faces
- Confirm attendance rows are added in `Attendance.csv`

## Notes

- `requirements.txt` uses standard `face_recognition` + `opencv-python`.
- `packages.txt` is not needed for Render Docker deployment.
- App logic/UI remains unchanged: camera capture, student registration, attendance CSV, and annotations all stay the same.
