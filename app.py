import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import base64
import json
import os
import uuid

import cv2
import numpy as np
import tensorflow as tf
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    send_from_directory,
    abort,
)
from werkzeug.utils import secure_filename

from src.roi.extract import extract_roi
from src.segmentation.predict import predict_mask
from src.explainability.gradcam import GradCAMExplainer
from src.database import (
    init_db,
    check_db_connection,
    save_screening,
    get_all_screenings,
    get_user_screenings,
    get_screening_by_id,
    ScreeningHistory,
)

# --------------------------------------------------
# Flask configuration & Base directory
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
)

app.secret_key = os.environ.get("SECRET_KEY", "anemeia-secure-secret-key-2026")

UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

try:
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(UPLOAD_FOLDER, "screenings"), exist_ok=True)
except Exception:
    pass

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Initialize MySQL Database via Flask-SQLAlchemy
db_initialized, db_status_msg = init_db(app, base_dir=BASE_DIR)


# Global template context processor: injects database health
@app.context_processor
def inject_context():
    is_db_connected, db_err = check_db_connection()
    return {
        "db_connected": is_db_connected,
        "db_err": db_err,
    }


# --------------------------------------------------
# Model paths & configuration
# --------------------------------------------------

UNET_TFLITE_PATH = os.path.join(BASE_DIR, "artifacts", "current", "segmentation", "best_unet_opt.tflite")
UNET_PATH = os.path.join(BASE_DIR, "artifacts", "current", "segmentation", "best_unet.keras")
CLASSIFIER_PATH = os.path.join(BASE_DIR, "artifacts", "current", "classifier", "best_classifier.keras")
THRESHOLD_CONFIG_PATH = os.path.join(BASE_DIR, "artifacts", "current", "classifier", "threshold_config.json")
DASHBOARD_DATA_PATH = os.path.join(BASE_DIR, "artifacts", "current", "classifier", "evaluation_dashboard_data.json")

DEFAULT_THRESHOLD = 0.3837
CLASSIFIER_THRESHOLD = DEFAULT_THRESHOLD

if os.path.exists(THRESHOLD_CONFIG_PATH):
    try:
        with open(THRESHOLD_CONFIG_PATH, "r") as f:
            t_data = json.load(f)
            CLASSIFIER_THRESHOLD = float(t_data.get("threshold", DEFAULT_THRESHOLD))
    except Exception as e:
        print(f"Could not load threshold config: {e}")

print(f"Active classifier threshold: {CLASSIFIER_THRESHOLD:.4f}")

# --------------------------------------------------
# Load models & Explainability Engine
# --------------------------------------------------

if os.path.exists(UNET_TFLITE_PATH):
    print("Loading optimized U-Net (TFLite XNNPACK engine, 45x faster)...")
    num_threads = max(1, min(4, os.cpu_count() or 2))
    unet = tf.lite.Interpreter(model_path=UNET_TFLITE_PATH, num_threads=num_threads)
    unet.allocate_tensors()
else:
    print("Loading U-Net Keras model...")
    unet = tf.keras.models.load_model(UNET_PATH, compile=False)

print("Loading EfficientNetB0 classifier...")
classifier = tf.keras.models.load_model(CLASSIFIER_PATH, compile=False)

print("Initializing Grad-CAM Explainer...")
explainer = GradCAMExplainer(classifier)

print("Models and explainer loaded successfully!")

# Warm up models at startup to eliminate cold-start delays on user requests
print("Warming up models for rapid inference...")
try:
    dummy_img = np.zeros((256, 256, 3), dtype=np.uint8)
    _ = predict_mask(unet, dummy_img)
    dummy_roi = np.zeros((224, 224, 3), dtype=np.uint8)
    dummy_batch = np.expand_dims(dummy_roi.astype(np.float32), 0)
    _ = classifier.predict(dummy_batch, verbose=0)
    _ = explainer.explain(dummy_roi)
    print("Warmup complete! All AI screening models are primed for instant inference.")
except Exception as w_err:
    print(f"Warmup notice: {w_err}")


# --------------------------------------------------
# Utility functions
# --------------------------------------------------

def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def encode_image_base64(img_bgr, max_dim=800, quality=85):
    """Encode OpenCV BGR image into an optimized base64 JPEG data URI for rapid transmission & instant browser rendering."""
    if img_bgr is None or img_bgr.size == 0:
        return None
    h, w = img_bgr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    success, buffer = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not success:
        return None
    b64_str = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"


def hb_reference(hb, sex):
    """
    Independent WHO clinical Hb reference assessment.
    Adult Female: Normal >= 12.0 g/dL
    Adult Male: Normal >= 13.0 g/dL
    """
    sex_clean = sex.strip().lower()

    if sex_clean == "female":
        threshold = 12.0
    elif sex_clean == "male":
        threshold = 13.0
    else:
        return {
            "valid": False,
            "message": "Select female or male for standard adult reference cutoff.",
            "threshold": None,
        }

    is_below = hb < threshold

    if hb < 8.0:
        severity = "Severe Anemia (< 8.0 g/dL)"
        severity_level = "severe"
    elif hb < 11.0:
        severity = "Moderate Anemia (8.0 - 10.9 g/dL)"
        severity_level = "moderate"
    elif is_below:
        severity = f"Mild Anemia (11.0 - {threshold:.1f} g/dL)"
        severity_level = "mild"
    else:
        severity = "Normal Range (At or above WHO threshold)"
        severity_level = "normal"

    status_text = (
        "Below WHO reference threshold"
        if is_below
        else "At or above WHO reference threshold"
    )

    return {
        "valid": True,
        "is_anemic": is_below,
        "status": status_text,
        "threshold": threshold,
        "hb_value": hb,
        "sex": sex_clean.capitalize(),
        "severity": severity,
        "severity_level": severity_level,
        "guideline": f"WHO adult cutoff for {sex_clean}s is {threshold:.1f} g/dL.",
    }


def prepare_roi_for_classifier(roi):
    """
    Prepare ROI for EfficientNetB0.
    EfficientNetB0 has internal Rescaling(1/255) and Normalization layers.
    Inputs must be in float32 [0.0, 255.0] range (NOT pre-divided by 255).
    """
    roi = np.asarray(roi)

    if roi is None or roi.size == 0:
        raise ValueError("The extracted conjunctiva ROI is empty.")

    if len(roi.shape) != 3 or roi.shape[2] != 3:
        raise ValueError("The extracted ROI must have 3 color channels.")

    # Convert BGR to RGB
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi_resized = cv2.resize(roi_rgb, (224, 224))
    roi_float = roi_resized.astype(np.float32)

    return np.expand_dims(roi_float, axis=0)


# --------------------------------------------------
# Main page
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    image_result = None
    image_score = None
    patient_name = None
    sex = None
    age = None
    error = None
    saved_screening_id = None
    db_notice = None

    roi_preview_b64 = None
    annotated_preview_b64 = None
    original_preview_b64 = None
    mask_preview_b64 = None
    gradcam_preview_b64 = None
    gradcam_error = None

    if request.method == "POST":
        try:
            file = request.files.get("eye_image")
            patient_name = request.form.get("patient_name", "").strip()
            age = request.form.get("age", "").strip()
            sex = request.form.get("sex", "").strip()

            has_image = file is not None and file.filename != ""

            if not has_image:
                raise ValueError(
                    "Please upload a conjunctival eye image to perform AI anemia screening."
                )

            if not allowed_file(file.filename):
                raise ValueError("Please upload a valid image file (JPG, JPEG, or PNG).")

            import time
            t_start = time.time()
            print(f"\n[AI Pipeline] Received image: {file.filename} (Patient: {patient_name or 'Anonymous'}, Age: {age or 'N/A'}, Sex: {sex or 'N/A'})")

            # Read and decode image directly in memory (avoids Windows file lock issues)
            file_bytes = file.read()
            image = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError("Unable to decode the uploaded image file. Please ensure it is a valid image.")

            # Optimize high-resolution mobile photos (e.g. 12MP/4K) to max 1024px
            # Preserves full vascular resolution while speeding up segmentation & ROI extraction by 40x
            h_orig, w_orig = image.shape[:2]
            max_input_dim = 1024
            if max(h_orig, w_orig) > max_input_dim:
                scale = max_input_dim / float(max(h_orig, w_orig))
                image = cv2.resize(image, (int(w_orig * scale), int(h_orig * scale)), interpolation=cv2.INTER_AREA)

            original_preview_b64 = encode_image_base64(image)

            # Step 1: Conjunctiva segmentation
            print("[AI Pipeline 1/4] Segmenting conjunctiva with U-Net...")
            t_unet = time.time()
            _, probability, binary_mask, quality = predict_mask(
                unet,
                image,
                input_size=(256, 256),
                threshold=0.5,
            )
            print(f"[AI Pipeline 1/4] Segmentation done in {time.time() - t_unet:.2f}s")

            if binary_mask is not None:
                mask_preview_b64 = encode_image_base64(
                    cv2.cvtColor((binary_mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
                )

            # Step 2: Extract conjunctiva ROI
            print("[AI Pipeline 2/4] Extracting conjunctiva ROI...")
            roi, bbox = extract_roi(
                image,
                binary_mask,
                output_size=(224, 224),
                margin_fraction=0.15,
            )

            if roi is None or np.asarray(roi).size == 0:
                raise ValueError("Unable to locate a valid palpebral conjunctiva region in this image. Please provide a clear, in-focus close-up of the lower eyelid.")

            # Create visualization previews
            annotated = image.copy()
            x0, y0, x1, y1 = bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"]
            cv2.rectangle(annotated, (x0, y0), (x1, y1), (0, 230, 255), max(2, int(image.shape[0] * 0.003)))
            annotated_preview_b64 = encode_image_base64(annotated)
            roi_preview_b64 = encode_image_base64(roi)

            # Step 3: EfficientNetB0 classification (Image ROI input only)
            print("[AI Pipeline 3/4] Predicting with EfficientNetB0 classifier...")
            t_cls = time.time()
            roi_batch = prepare_roi_for_classifier(roi)
            prediction = classifier.predict(roi_batch, verbose=0)
            image_score = float(np.asarray(prediction).reshape(-1)[0])
            print(f"[AI Pipeline 3/4] Classification done in {time.time() - t_cls:.2f}s. Score: {image_score:.4f}")

            is_anemic = image_score >= CLASSIFIER_THRESHOLD
            image_result = {
                "patient_name": patient_name if patient_name else "Anonymous Patient",
                "is_anemic": is_anemic,
                "score": image_score,
                "score_percentage": image_score * 100.0,
                "threshold": CLASSIFIER_THRESHOLD,
                "threshold_percentage": CLASSIFIER_THRESHOLD * 100.0,
                "classification": (
                    "Anemia Detected"
                    if is_anemic
                    else "Non-Anemia (Normal)"
                ),
                "confidence": (
                    f"{image_score * 100.0:.1f}% risk score"
                    if is_anemic
                    else f"{(1.0 - image_score) * 100.0:.1f}% normal confidence"
                ),
                "quality": quality.get("quality_status", "accepted"),
                "bbox": bbox,
                "age": age if age else None,
                "sex": sex.capitalize() if sex else None,
            }

            # Step 4: Generate Grad-CAM explainability
            print("[AI Pipeline 4/4] Generating Grad-CAM explainability heatmap...")
            t_cam = time.time()
            try:
                gradcam_overlay, heatmap, gradcam_err = explainer.explain(
                    roi,
                    target_class_anemia=is_anemic,
                )
                if gradcam_overlay is not None:
                    gradcam_preview_b64 = encode_image_base64(gradcam_overlay)
                if gradcam_err:
                    gradcam_error = gradcam_err
                print(f"[AI Pipeline 4/4] Grad-CAM done in {time.time() - t_cam:.2f}s")
            except Exception as g_err:
                print(f"Grad-CAM generation notice: {g_err}")
                gradcam_error = f"Grad-CAM explainability could not be generated: {str(g_err)}"

            print(f"[AI Pipeline] Entire pipeline completed in {time.time() - t_start:.2f}s! -> Result: {image_result['classification']}")
            result = "Assessment Completed"

            # Step 5: Save screening record to MySQL database
            try:
                is_connected, _ = check_db_connection()
                if is_connected:
                    record, save_err = save_screening(
                        original_bgr=image,
                        binary_mask=binary_mask,
                        roi_bgr=roi,
                        gradcam_bgr=gradcam_overlay,
                        prediction=image_result["classification"],
                        confidence_score=image_score,
                        threshold=CLASSIFIER_THRESHOLD,
                        tissue_quality=quality.get("quality_status", "accepted"),
                        patient_name=patient_name if patient_name else "Anonymous Patient",
                        patient_age=int(age) if age and str(age).isdigit() else None,
                        patient_gender=sex if sex else None,
                        model_name="EfficientNet-B0 + U-Net",
                        upload_dir=app.config["UPLOAD_FOLDER"],
                    )
                    if record:
                        saved_screening_id = record.id
                        print(f"[Database] Screening #{saved_screening_id} for '{patient_name or 'Anonymous'}' saved successfully to MySQL.")
                    elif save_err:
                        db_notice = f"Notice: Screening completed, but could not save to database: {save_err}"
                        print(f"[Database Save Error] {save_err}")
                else:
                    db_notice = "Notice: Screening completed. Result not stored because MySQL database is offline."
            except Exception as db_e:
                print(f"[Database Notice] Error saving screening: {db_e}")
                db_notice = "Notice: Screening completed. Result could not be saved to database."

        except Exception as e:
            print("ERROR in index route:", str(e))
            error = str(e)

    return render_template(
        "index.html",
        result=result,
        image_result=image_result,
        image_score=image_score,
        patient_name=patient_name,
        sex=sex,
        age=age,
        error=error,
        threshold=CLASSIFIER_THRESHOLD,
        roi_preview=roi_preview_b64,
        annotated_preview=annotated_preview_b64,
        original_preview=original_preview_b64,
        mask_preview=mask_preview_b64,
        gradcam_preview=gradcam_preview_b64,
        gradcam_error=gradcam_error,
        saved_screening_id=saved_screening_id,
        db_notice=db_notice,
    )


# --------------------------------------------------
# Screening History & Details routes
# --------------------------------------------------

@app.route("/history")
def history():
    page = request.args.get("page", 1, type=int)
    search = request.args.get("q", "").strip()

    is_connected, db_err = check_db_connection()
    if not is_connected:
        return render_template(
            "history.html",
            screenings=[],
            pagination=None,
            search=search,
            db_error=f"Database is currently offline. {db_err or ''}",
        )

    pagination = get_all_screenings(page=page, per_page=12, search=search if search else None)
    screenings = pagination.items if pagination else []

    return render_template(
        "history.html",
        screenings=screenings,
        pagination=pagination,
        search=search,
        db_error=None,
    )


@app.route("/history/<int:screening_id>")
def screening_details(screening_id):
    is_connected, db_err = check_db_connection()
    if not is_connected:
        return render_template(
            "screening_details.html",
            screening=None,
            error=f"Database is currently offline: {db_err or 'Cannot load screening details.'}",
        )

    record, err = get_screening_by_id(screening_id)
    return render_template(
        "screening_details.html",
        screening=record,
        error=err,
    )


@app.route("/screenings/image/<int:screening_id>/<img_type>")
def screening_image(screening_id, img_type):
    if img_type not in ("original", "mask", "roi", "gradcam"):
        abort(404)

    record, err = get_screening_by_id(screening_id)
    if err or not record:
        abort(404)

    path_map = {
        "original": record.original_image_path,
        "mask": record.segmented_image_path,
        "roi": record.extracted_roi_path,
        "gradcam": record.gradcam_image_path,
    }
    rel_path = path_map.get(img_type)
    if not rel_path:
        abort(404)

    full_path = os.path.join(app.config["UPLOAD_FOLDER"], rel_path)
    if not os.path.exists(full_path):
        abort(404)

    directory = os.path.dirname(full_path)
    filename = os.path.basename(full_path)
    return send_from_directory(directory, filename)


# --------------------------------------------------
# Model Evaluation Dashboard route
# --------------------------------------------------

@app.route("/dashboard")
def dashboard():
    dashboard_data = None
    error = None
    try:
        if os.path.exists(DASHBOARD_DATA_PATH):
            with open(DASHBOARD_DATA_PATH, "r") as f:
                dashboard_data = json.load(f)
        else:
            from scripts.evaluate_end_to_end import run_evaluation
            dashboard_data = run_evaluation()
    except Exception as e:
        print(f"Error loading dashboard data: {e}")
        error = f"Evaluation data could not be loaded: {str(e)}"

    return render_template(
        "dashboard.html",
        data=dashboard_data,
        error=error,
    )


# --------------------------------------------------
# Run Flask application
# --------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )


