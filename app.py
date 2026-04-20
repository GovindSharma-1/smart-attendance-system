import io
import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw

# We import face_recognition safely so the app can show a user-friendly
# message instead of crashing if dependency resolution fails.
try:
    import face_recognition
    FACE_LIB_READY = True
    FACE_LIB_ERROR = ""
except Exception as e:
    face_recognition = None
    FACE_LIB_READY = False
    FACE_LIB_ERROR = str(e)


# ----------------------------- #
# Project-level file/folder paths
# ----------------------------- #
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STUDENT_IMAGES_DIR = os.path.join(BASE_DIR, "student_images")
ENCODINGS_PATH = os.path.join(BASE_DIR, "encodings.pkl")
ATTENDANCE_PATH = os.path.join(BASE_DIR, "Attendance.csv")


# ----------------------------- #
# Helper: Create required folders/files safely
# ----------------------------- #
def ensure_project_files() -> None:
    """
    Ensures all mandatory project assets exist.
    This avoids runtime errors in cloud or local setup.
    """
    os.makedirs(STUDENT_IMAGES_DIR, exist_ok=True)

    # Create empty attendance CSV with expected schema if not present.
    if not os.path.exists(ATTENDANCE_PATH):
        df = pd.DataFrame(columns=["Name", "Time", "Date"])
        df.to_csv(ATTENDANCE_PATH, index=False)


# ----------------------------- #
# Helper: Load known encodings from disk
# ----------------------------- #
def load_encodings_from_pickle() -> dict:
    """
    Returns a dict with:
    {
      "names": [str, ...],
      "encodings": [np.ndarray, ...]
    }
    """
    if not os.path.exists(ENCODINGS_PATH):
        return {"names": [], "encodings": []}

    try:
        with open(ENCODINGS_PATH, "rb") as f:
            data = pickle.load(f)

        # Defensive checks for corrupted/incompatible pickle.
        if not isinstance(data, dict):
            return {"names": [], "encodings": []}
        if "names" not in data or "encodings" not in data:
            return {"names": [], "encodings": []}
        return data
    except Exception:
        return {"names": [], "encodings": []}


# ----------------------------- #
# Helper: Persist encodings to disk
# ----------------------------- #
def save_encodings_to_pickle(data: dict) -> None:
    with open(ENCODINGS_PATH, "wb") as f:
        pickle.dump(data, f)


# ----------------------------- #
# Helper: Convert uploaded file bytes -> PIL image -> numpy RGB
# ----------------------------- #
def read_image_to_rgb_array(file_bytes: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    return np.array(image)


# ----------------------------- #
# Helper: Extract one face encoding from image
# ----------------------------- #
def extract_single_face_encoding(rgb_array: np.ndarray):
    """
    Returns the first face encoding if found.
    Returns None if no face is detected.
    """
    if not FACE_LIB_READY:
        return None

    encodings = face_recognition.face_encodings(rgb_array)
    if len(encodings) == 0:
        return None
    return encodings[0]


# ----------------------------- #
# Helper: Learn faces from uploaded files
# ----------------------------- #
def register_students_from_uploads(uploaded_files):
    """
    For each file:
    - Name is derived from filename (Govind.jpg -> GOVIND)
    - Image is saved in student_images/
    - Face encoding is generated and stored
    """
    if not uploaded_files:
        st.info("Upload one or more student images to register faces.")
        return

    known_data = st.session_state.known_data
    success_count = 0
    skipped_count = 0

    for file in uploaded_files:
        try:
            raw_bytes = file.read()
            rgb_array = read_image_to_rgb_array(raw_bytes)

            # Name extraction from filename (without extension).
            student_name = os.path.splitext(file.name)[0].strip().upper()
            if not student_name:
                skipped_count += 1
                continue

            # Save original file for record/future retraining.
            save_path = os.path.join(STUDENT_IMAGES_DIR, file.name)
            with open(save_path, "wb") as out:
                out.write(raw_bytes)

            encoding = extract_single_face_encoding(rgb_array)
            if encoding is None:
                st.warning(f"No clear face found in `{file.name}`. Skipped.")
                skipped_count += 1
                continue

            # If same student already exists, replace old encoding with latest.
            if student_name in known_data["names"]:
                idx = known_data["names"].index(student_name)
                known_data["encodings"][idx] = encoding
            else:
                known_data["names"].append(student_name)
                known_data["encodings"].append(encoding)

            success_count += 1
        except Exception as e:
            skipped_count += 1
            st.error(f"Error processing `{file.name}`: {e}")

    save_encodings_to_pickle(known_data)
    st.session_state.known_data = known_data
    st.success(
        f"Registration complete. Added/updated {success_count} students, skipped {skipped_count}."
    )


# ----------------------------- #
# Helper: Learn faces from existing images folder
# ----------------------------- #
def register_students_from_folder():
    """
    Reads student_images/ and trains encodings from files that exist already.
    """
    valid_ext = (".jpg", ".jpeg", ".png")
    files = [
        f
        for f in os.listdir(STUDENT_IMAGES_DIR)
        if os.path.isfile(os.path.join(STUDENT_IMAGES_DIR, f))
        and f.lower().endswith(valid_ext)
    ]

    if not files:
        st.warning("`student_images/` is empty. Upload student photos first.")
        return

    known_data = st.session_state.known_data
    trained = 0
    skipped = 0

    for filename in files:
        try:
            file_path = os.path.join(STUDENT_IMAGES_DIR, filename)
            with open(file_path, "rb") as f:
                rgb_array = read_image_to_rgb_array(f.read())

            student_name = os.path.splitext(filename)[0].strip().upper()
            encoding = extract_single_face_encoding(rgb_array)
            if encoding is None:
                skipped += 1
                continue

            if student_name in known_data["names"]:
                idx = known_data["names"].index(student_name)
                known_data["encodings"][idx] = encoding
            else:
                known_data["names"].append(student_name)
                known_data["encodings"].append(encoding)

            trained += 1
        except Exception:
            skipped += 1

    save_encodings_to_pickle(known_data)
    st.session_state.known_data = known_data
    st.success(
        f"Folder training done. Added/updated {trained} students, skipped {skipped}."
    )


# ----------------------------- #
# Helper: Mark attendance once per student per day
# ----------------------------- #
def mark_attendance_once_per_day(name: str) -> bool:
    """
    Returns True if new attendance row was added.
    Returns False if already marked today.
    """
    df = pd.read_csv(ATTENDANCE_PATH)
    today = datetime.now().strftime("%Y-%m-%d")

    already_marked = ((df["Name"] == name) & (df["Date"] == today)).any()
    if already_marked:
        return False

    now = datetime.now()
    new_row = {
        "Name": name,
        "Time": now.strftime("%H:%M:%S"),
        "Date": now.strftime("%Y-%m-%d"),
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(ATTENDANCE_PATH, index=False)
    return True


# ----------------------------- #
# Helper: Detect and recognize faces + annotate image
# ----------------------------- #
def recognize_and_annotate(pil_image: Image.Image):
    """
    Detects all faces, predicts names, and draws green boxes + labels.
    Returns:
      - annotated PIL image
      - list of recognized names
    """
    if not FACE_LIB_READY:
        return pil_image, []

    rgb_array = np.array(pil_image.convert("RGB"))
    face_locations = face_recognition.face_locations(rgb_array)
    face_encodings = face_recognition.face_encodings(rgb_array, face_locations)

    known_names = st.session_state.known_data["names"]
    known_encodings = st.session_state.known_data["encodings"]

    annotated = pil_image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    recognized_names = []

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        predicted_name = "Unknown"

        if len(known_encodings) > 0:
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.5)
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            best_idx = int(np.argmin(distances)) if len(distances) > 0 else -1

            if best_idx >= 0 and matches[best_idx]:
                predicted_name = known_names[best_idx]

        # Draw green rectangle and label text.
        draw.rectangle([(left, top), (right, bottom)], outline=(0, 255, 0), width=3)
        draw.rectangle([(left, bottom - 25), (right, bottom)], fill=(0, 255, 0))
        draw.text((left + 6, bottom - 22), predicted_name, fill=(0, 0, 0))

        if predicted_name != "Unknown":
            recognized_names.append(predicted_name)

    return annotated, recognized_names


# ----------------------------- #
# Streamlit page setup
# ----------------------------- #
st.set_page_config(page_title="Smart Attendance System", page_icon="🚀", layout="wide")

# A little custom CSS for a cleaner modern appearance.
st.markdown(
    """
    <style>
      .main-title {font-size: 2.2rem; font-weight: 700; margin-bottom: 0.2rem;}
      .subtitle {color: #6c757d; margin-bottom: 1rem;}
      div[data-testid="stSidebar"] {background: linear-gradient(180deg, #f8f9fa 0%, #eef4ff 100%);}
    </style>
    """,
    unsafe_allow_html=True,
)

# Stop early with clear setup guidance when face library is unavailable.
if not FACE_LIB_READY:
    st.error("Face recognition dependency is not ready.")
    st.code(
        "pip install -r requirements.txt\n"
        "pip install git+https://github.com/ageitgey/face_recognition_models"
    )
    st.caption(f"Technical detail: {FACE_LIB_ERROR}")
    st.stop()

st.markdown('<div class="main-title">🚀 Smart Attendance System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">AI-powered face recognition attendance for your portfolio project</div>',
    unsafe_allow_html=True,
)


# Initialize files/folders and session cache once.
ensure_project_files()
if "known_data" not in st.session_state:
    st.session_state.known_data = load_encodings_from_pickle()
if "last_recognized" not in st.session_state:
    st.session_state.last_recognized = []


# ----------------------------- #
# Sidebar: registration + attendance download
# ----------------------------- #
st.sidebar.title("Control Panel")
st.sidebar.subheader("Register Students")
uploaded_students = st.sidebar.file_uploader(
    "Upload student photos (example: Govind.jpg)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
)
if st.sidebar.button("Register Uploaded Students"):
    register_students_from_uploads(uploaded_students)

if st.sidebar.button("Train From student_images Folder"):
    register_students_from_folder()

st.sidebar.markdown("---")
st.sidebar.subheader("View Attendance")

try:
    with open(ATTENDANCE_PATH, "rb") as f:
        st.sidebar.download_button(
            "Download Attendance.csv",
            data=f,
            file_name="Attendance.csv",
            mime="text/csv",
        )
except Exception:
    st.sidebar.info("Attendance file will be available after first successful mark.")

st.sidebar.markdown("---")
st.sidebar.caption(f"Registered students: {len(st.session_state.known_data['names'])}")


# ----------------------------- #
# Main: capture/upload image and run recognition
# ----------------------------- #
st.subheader("Mark Attendance")
st.write("Use camera capture (recommended) or upload an image as fallback.")

camera_file = st.camera_input("Live Camera Capture (click the capture button in camera widget)")
uploaded_test_image = st.file_uploader(
    "Fallback: Upload class photo/selfie",
    type=["jpg", "jpeg", "png"],
    key="test_image_uploader",
)

process_clicked = st.button("Capture / Process Attendance", type="primary")

if process_clicked:
    if len(st.session_state.known_data["encodings"]) == 0:
        st.error("No student encodings found. Register students first from the sidebar.")
    else:
        selected_bytes = None
        source_label = None

        if camera_file is not None:
            selected_bytes = camera_file.getvalue()
            source_label = "camera"
        elif uploaded_test_image is not None:
            selected_bytes = uploaded_test_image.read()
            source_label = "uploaded image"

        if selected_bytes is None:
            st.warning("Please capture an image from camera or upload a fallback image.")
        else:
            try:
                input_image = Image.open(io.BytesIO(selected_bytes)).convert("RGB")
                annotated_image, recognized_names = recognize_and_annotate(input_image)
                st.image(
                    annotated_image,
                    caption=f"Processed from {source_label} with recognized faces",
                    use_container_width=True,
                )

                if len(recognized_names) == 0:
                    st.info("No known face recognized in this image.")
                else:
                    unique_names = sorted(set(recognized_names))
                    st.session_state.last_recognized = unique_names

                    for name in unique_names:
                        if mark_attendance_once_per_day(name):
                            st.success(f"✅ Attendance marked for {name}")
                        else:
                            st.info(f"Already marked today for {name}")
            except Exception as e:
                st.error(f"Unable to process image. Check image quality/permissions. Details: {e}")


# ----------------------------- #
# Optional preview table on main page
# ----------------------------- #
st.markdown("---")
st.subheader("Today's Attendance Snapshot")
try:
    attendance_df = pd.read_csv(ATTENDANCE_PATH)
    today = datetime.now().strftime("%Y-%m-%d")
    today_df = attendance_df[attendance_df["Date"] == today]
    st.dataframe(today_df, use_container_width=True)
except Exception as e:
    st.warning(f"Could not load attendance table yet: {e}")

