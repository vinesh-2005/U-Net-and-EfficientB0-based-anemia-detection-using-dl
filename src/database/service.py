"""Database service functions for authentication, screening records, and safe disk storage."""
import logging
import os
import re
import uuid
import cv2
import numpy as np
from sqlalchemy import text
from .models import db, User, Patient, ScreeningHistory, utc_now

logger = logging.getLogger(__name__)

EMAIL_REGEX = re.compile(r"^[\w\.\+\-]+@[a-zA-Z0-9\-]+\.[a-zA-Z0-9\-\.]+$")


def check_db_connection(app=None) -> tuple[bool, str | None]:
    """
    Safely tests whether the configured database is reachable and accepting queries.
    Returns (True, None) if healthy, or (False, error_message) on failure.
    """
    try:
        with db.engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True, None
    except Exception as e:
        err_msg = str(e)
        # Avoid exposing raw passwords or internals in user-facing message
        safe_msg = "Database connection offline or unreachable."
        if "Access denied" in err_msg:
            safe_msg = "Database authentication failed (check DB_USER and DB_PASSWORD)."
        elif "Can't connect to MySQL server" in err_msg or "Connection refused" in err_msg:
            safe_msg = "MySQL server is not running or not reachable on the configured host/port."
        elif "Unknown database" in err_msg:
            safe_msg = "Database 'anemeia_db' does not exist yet. Please run schema.sql first."
        logger.warning(f"Database health check failed: {err_msg}")
        return False, safe_msg


def ensure_patient_schema_and_backfill():
    """
    Ensures that the `patient_id` column exists in `screening_history` and that all
    historical screening records are associated with a Patient profile.
    """
    try:
        # Check if patient_id column exists in screening_history
        inspector = db.inspect(db.engine)
        columns = [c["name"] for c in inspector.get_columns("screening_history")]
        if "patient_id" not in columns:
            logger.info("Adding missing column 'patient_id' to screening_history...")
            with db.engine.begin() as conn:
                conn.execute(text("ALTER TABLE screening_history ADD COLUMN patient_id VARCHAR(50)"))

        # Link any existing historical records without patient_id to a Patient profile
        orphans = ScreeningHistory.query.filter(
            (ScreeningHistory.patient_id.is_(None)) | (ScreeningHistory.patient_id == "")
        ).all()

        if orphans:
            for s in orphans:
                name = (s.patient_name or "").strip() or "Anonymous Patient"
                pat = get_or_create_patient(
                    patient_name=name,
                    age=s.patient_age,
                    gender=s.patient_gender,
                )
                s.patient_id = pat.id
            db.session.commit()
            logger.info(f"Successfully linked {len(orphans)} historical screenings to patient profiles.")
    except Exception as e:
        db.session.rollback()
        logger.warning(f"Notice during schema/patient migration: {e}")


def init_db(app, base_dir=None):
    """
    Initializes Flask-SQLAlchemy with the application and attempts safe table creation.
    If MySQL server is offline or not installed, automatically activates a resilient
    local SQLite database fallback (anemeia_db.sqlite3) so screening history is never lost.
    """
    from .config import get_db_uri
    db_uri = get_db_uri(base_dir)

    is_fallback = False
    if not db_uri.startswith("sqlite"):
        try:
            from sqlalchemy import create_engine
            # Quick probe to test if MySQL server is running and responding
            test_engine = create_engine(
                db_uri,
                connect_args={"connect_timeout": 2},
            )
            with test_engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            test_engine.dispose()
        except Exception as probe_err:
            logger.warning(
                f"MySQL server is offline or unreachable ({probe_err}). "
                "Activating resilient local SQLite database fallback."
            )
            sqlite_path = os.path.join(base_dir or ".", "anemeia_db.sqlite3")
            db_uri = f"sqlite:///{sqlite_path}"
            is_fallback = True

    app.config["SQLALCHEMY_DATABASE_URI"] = db_uri
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
        "pool_recycle": 280,
        "pool_pre_ping": True,
    }

    db.init_app(app)

    with app.app_context():
        try:
            db.create_all()
            ensure_patient_schema_and_backfill()
            if is_fallback:
                logger.info("Local SQLite database initialized successfully at anemeia_db.sqlite3")
                return True, "Connected (Local Storage Fallback)"
            else:
                logger.info("MySQL database connected and tables verified successfully.")
                return True, "Connected (MySQL)"
        except Exception as e:
            logger.warning(f"Could not initialize database tables: {e}")
            return False, str(e)


# --------------------------------------------------
# Patient Entity Management & Reuse
# --------------------------------------------------

def get_or_create_patient(
    patient_name: str,
    patient_id: str | None = None,
    age: int | None = None,
    gender: str | None = None,
) -> Patient:
    """
    Finds or creates a Patient entity.
    If the same patient visits again (by ID or matching name), reuses the existing Patient ID
    instead of creating a duplicate patient.
    """
    clean_name = (patient_name or "").strip()
    if not clean_name:
        clean_name = "Anonymous Patient"

    clean_id = (patient_id or "").strip()

    patient = None
    if clean_id:
        patient = db.session.get(Patient, clean_id)

    # If no match by ID and not an anonymous patient, look up by patient name (case-insensitive)
    if not patient and clean_name.lower() != "anonymous patient":
        patient = Patient.query.filter(Patient.name.ilike(clean_name)).first()

    if patient:
        # Update demographics if provided
        if age:
            patient.age = age
        if gender:
            patient.gender = gender.capitalize()
        patient.updated_at = utc_now()
    else:
        # Generate clean patient ID if not provided: PID-1001, PID-1002, etc.
        if not clean_id:
            total_patients = Patient.query.count()
            candidate_num = 1001 + total_patients
            while db.session.get(Patient, f"PID-{candidate_num}"):
                candidate_num += 1
            clean_id = f"PID-{candidate_num}"

        patient = Patient(
            id=clean_id,
            name=clean_name,
            age=age,
            gender=gender.capitalize() if gender else None,
        )
        db.session.add(patient)
        db.session.flush()

    return patient


# --------------------------------------------------
# User Management & Authentication (Optional/Admin)
# --------------------------------------------------

def register_user(full_name: str, email: str, password: str, phone_number: str = None) -> tuple[User | None, str | None]:
    """
    Registers a new user with validation and secure password hashing.
    Returns (user, None) on success, or (None, error_message) on failure.
    """
    full_name = (full_name or "").strip()
    email = (email or "").strip().lower()
    password = password or ""
    phone_number = (phone_number or "").strip() if phone_number else None

    if not full_name or len(full_name) < 2:
        return None, "Please provide your full name (minimum 2 characters)."

    if not email or not EMAIL_REGEX.match(email):
        return None, "Please provide a valid email address."

    if not password or len(password) < 6:
        return None, "Password must be at least 6 characters long."

    try:
        # Check for existing email
        existing = User.query.filter_by(email=email).first()
        if existing:
            return None, "An account with this email address already exists. Please log in."

        user = User(
            full_name=full_name,
            email=email,
            phone_number=phone_number,
        )
        user.set_password(password)

        db.session.add(user)
        db.session.commit()
        return user, None
    except Exception as e:
        db.session.rollback()
        logger.exception("Error during user registration:")
        return None, f"Registration failed due to a database error: {str(e)}"


def authenticate_user(email: str, password: str) -> tuple[User | None, str | None]:
    """
    Authenticates a user against stored password hash.
    Returns (user, None) on success, or (None, error_message) on failure.
    """
    email = (email or "").strip().lower()
    password = password or ""

    if not email or not password:
        return None, "Please enter both email and password."

    try:
        user = User.query.filter_by(email=email).first()
        if not user or not user.check_password(password):
            return None, "Invalid email or password. Please try again."

        return user, None
    except Exception as e:
        logger.exception("Error during user authentication:")
        return None, f"Authentication error: {str(e)}"


# --------------------------------------------------
# Screening History & Disk Persistence
# --------------------------------------------------

def save_screening(
    original_bgr: np.ndarray,
    binary_mask: np.ndarray,
    roi_bgr: np.ndarray,
    gradcam_bgr: np.ndarray | None,
    prediction: str,
    confidence_score: float,
    threshold: float,
    tissue_quality: str = "accepted",
    patient_id: str | None = None,
    patient_name: str | None = None,
    patient_age: int | None = None,
    patient_gender: str | None = None,
    user_id: int | None = None,
    model_name: str = "EfficientNet-B0 + U-Net",
    upload_dir: str = None,
) -> tuple[ScreeningHistory | None, str | None]:
    """
    Persists screening images safely to disk and records metadata in the database.
    Associates multiple visits with the same Patient profile.
    Returns (ScreeningHistory, None) on success, or (None, error_message) on failure.
    """
    if original_bgr is None or roi_bgr is None:
        return None, "Cannot save screening: Missing image or ROI data."

    if not upload_dir:
        return None, "Upload directory not configured."

    # Generate a unique storage directory for this specific screening
    screening_uuid = uuid.uuid4().hex
    target_dir = os.path.join(upload_dir, "screenings", screening_uuid)
    os.makedirs(target_dir, exist_ok=True)

    # File names
    orig_rel = f"screenings/{screening_uuid}/original.jpg"
    mask_rel = f"screenings/{screening_uuid}/mask.png"
    roi_rel = f"screenings/{screening_uuid}/roi.jpg"
    gradcam_rel = f"screenings/{screening_uuid}/gradcam.jpg" if gradcam_bgr is not None else None

    clean_name = (patient_name or "").strip()
    if not clean_name:
        clean_name = "Anonymous Patient"

    try:
        # Retrieve or create Patient profile (reusing existing Patient ID on repeat visits)
        patient = get_or_create_patient(
            patient_name=clean_name,
            patient_id=patient_id,
            age=patient_age,
            gender=patient_gender,
        )

        # Save images to disk
        cv2.imwrite(os.path.join(target_dir, "original.jpg"), original_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        if binary_mask is not None:
            mask_uint8 = (binary_mask * 255).astype(np.uint8) if binary_mask.max() <= 1 else binary_mask.astype(np.uint8)
            cv2.imwrite(os.path.join(target_dir, "mask.png"), mask_uint8)
        else:
            cv2.imwrite(os.path.join(target_dir, "mask.png"), np.zeros((224, 224), dtype=np.uint8))

        cv2.imwrite(os.path.join(target_dir, "roi.jpg"), roi_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        if gradcam_bgr is not None:
            cv2.imwrite(os.path.join(target_dir, "gradcam.jpg"), gradcam_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

        # Create database record
        record = ScreeningHistory(
            patient_id=patient.id,
            patient_name=patient.name,
            patient_age=patient_age if patient_age else patient.age,
            patient_gender=patient_gender.capitalize() if patient_gender else patient.gender,
            user_id=user_id,
            original_image_path=orig_rel,
            segmented_image_path=mask_rel,
            extracted_roi_path=roi_rel,
            gradcam_image_path=gradcam_rel,
            prediction=prediction,
            confidence_score=float(confidence_score),
            threshold=float(threshold),
            model_name=model_name,
            tissue_quality=tissue_quality,
        )

        db.session.add(record)
        db.session.commit()
        return record, None

    except Exception as e:
        db.session.rollback()
        logger.exception("Failed to save screening record to database:")
        return None, f"Database error saving screening: {str(e)}"


def get_all_screenings(page: int = 1, per_page: int = 15, search: str | None = None):
    """
    Retrieves paginated screening history for all patients.
    Supports optional case-insensitive search by patient name or patient ID.
    """
    try:
        query = ScreeningHistory.query
        if search:
            term = f"%{search.strip()}%"
            query = query.filter(
                (ScreeningHistory.patient_name.ilike(term))
                | (ScreeningHistory.patient_id.ilike(term))
            )

        query = query.order_by(ScreeningHistory.created_at.desc())
        return query.paginate(page=page, per_page=per_page, error_out=False)
    except Exception as e:
        logger.exception("Error querying screening history:")
        return None


def get_user_screenings(user_id: int | None = None, page: int = 1, per_page: int = 15, search: str | None = None):
    """Backwards-compatible alias for get_all_screenings."""
    return get_all_screenings(page=page, per_page=per_page, search=search)


def get_screening_by_id(screening_id: int, user_id: int | None = None) -> tuple[ScreeningHistory | None, str | None]:
    """
    Retrieves a single screening record by primary key.
    """
    try:
        record = db.session.get(ScreeningHistory, screening_id)
        if not record:
            return None, "Screening record not found."
        return record, None
    except Exception as e:
        logger.exception("Error retrieving screening record:")
        return None, f"Database error: {str(e)}"


# --------------------------------------------------
# Longitudinal Patient Anemia Progress Tracking
# --------------------------------------------------

def get_patient_progress_data(
    screening_id: int | None = None,
    patient_id: str | None = None,
) -> dict | None:
    """
    Computes longitudinal progress comparison, trend metrics, chart series, and
    timeline data for a patient.
    Can be queried either by specific screening_id or patient_id.
    """
    try:
        target_screening = None
        if screening_id:
            target_screening = db.session.get(ScreeningHistory, screening_id)

        patient = None
        if target_screening and target_screening.patient_id:
            patient = db.session.get(Patient, target_screening.patient_id)

        if not patient and target_screening and target_screening.patient_name:
            patient = Patient.query.filter(Patient.name.ilike(target_screening.patient_name)).first()

        if not patient and patient_id:
            patient = db.session.get(Patient, patient_id)

        # If patient profile not explicitly found, build a profile from screening
        if not patient and target_screening:
            patient = get_or_create_patient(
                patient_name=target_screening.patient_name or "Anonymous Patient",
                patient_id=target_screening.patient_id,
                age=target_screening.patient_age,
                gender=target_screening.patient_gender,
            )
            target_screening.patient_id = patient.id
            db.session.commit()

        if not patient and not target_screening:
            return None

        # Fetch all chronological screenings for this patient
        query = ScreeningHistory.query
        if patient:
            query = query.filter(
                (ScreeningHistory.patient_id == patient.id)
                | (ScreeningHistory.patient_name.ilike(patient.name))
            )
        elif target_screening:
            query = query.filter(ScreeningHistory.id == target_screening.id)

        all_screenings = query.order_by(ScreeningHistory.created_at.asc()).all()

        if not all_screenings and target_screening:
            all_screenings = [target_screening]

        # Determine current screening and previous screening
        if target_screening:
            current_screening = target_screening
            cur_idx = -1
            for idx, sc in enumerate(all_screenings):
                if sc.id == target_screening.id:
                    cur_idx = idx
                    break
            previous_screening = all_screenings[cur_idx - 1] if cur_idx > 0 else None
        else:
            current_screening = all_screenings[-1]
            previous_screening = all_screenings[-2] if len(all_screenings) >= 2 else None

        current_score = current_screening.confidence_percentage
        previous_score = previous_screening.confidence_percentage if previous_screening else None

        if previous_score is not None:
            diff_points = round(current_score - previous_score, 1)
            if diff_points < 0:
                trend_direction = "decreased"
                trend_status = "Risk Score Decreased"
                trend_icon = "↓"
                trend_color = "#059669"
                trend_badge_class = "trend-decreased"
                trend_description = (
                    f"Model-predicted risk score decreased by {abs(diff_points):.1f} percentage points "
                    f"(from {previous_score:.1f}% to {current_score:.1f}%)."
                )
            elif diff_points > 0:
                trend_direction = "increased"
                trend_status = "Risk Score Increased"
                trend_icon = "↑"
                trend_color = "#DC2626"
                trend_badge_class = "trend-increased"
                trend_description = (
                    f"Model-predicted risk score increased by {abs(diff_points):.1f} percentage points "
                    f"(from {previous_score:.1f}% to {current_score:.1f}%)."
                )
            else:
                trend_direction = "unchanged"
                trend_status = "No Change"
                trend_icon = "→"
                trend_color = "#64748B"
                trend_badge_class = "trend-unchanged"
                trend_description = (
                    f"Model-predicted risk score remained unchanged at {current_score:.1f}%."
                )
        else:
            diff_points = None
            trend_direction = "baseline"
            trend_status = "Initial Baseline Established"
            trend_icon = "●"
            trend_color = "#0F766E"
            trend_badge_class = "trend-baseline"
            trend_description = (
                "First screening on record. Longitudinal trend tracking will compare against this baseline."
            )

        # Prepare chart series data
        chart_labels = []
        chart_scores = []
        chart_predictions = []
        chart_ids = []
        chart_tooltips = []

        timeline_records = []
        for s in all_screenings:
            date_str = s.created_at.strftime("%b %d, %Y") if s.created_at else "N/A"
            time_str = s.created_at.strftime("%I:%M %p") if s.created_at else ""
            full_date = f"{date_str} at {time_str}" if time_str else date_str

            chart_labels.append(date_str)
            chart_scores.append(s.confidence_percentage)
            chart_predictions.append(s.prediction)
            chart_ids.append(s.id)
            chart_tooltips.append(f"{full_date}: {s.confidence_percentage:.1f}% ({s.prediction})")

            timeline_records.append({
                "id": s.id,
                "date_formatted": date_str,
                "time_formatted": time_str,
                "created_at_str": full_date,
                "is_anemic": s.is_anemic,
                "prediction": s.prediction,
                "confidence_score": s.confidence_score,
                "confidence_percentage": s.confidence_percentage,
                "threshold_percentage": s.threshold_percentage,
                "tissue_quality": s.tissue_quality or "accepted",
                "is_active": (s.id == current_screening.id),
                "original_image_path": s.original_image_path,
                "segmented_image_path": s.segmented_image_path,
                "extracted_roi_path": s.extracted_roi_path,
                "gradcam_image_path": s.gradcam_image_path,
            })

        patient_dict = {
            "id": patient.id if patient else (target_screening.display_patient_id if target_screening else "PID-UNKNOWN"),
            "name": patient.name if patient else (target_screening.patient_name if target_screening else "Anonymous Patient"),
            "age": patient.age if patient else (target_screening.patient_age if target_screening else None),
            "gender": patient.gender if patient else (target_screening.patient_gender if target_screening else None),
            "created_at": patient.created_at.strftime("%b %d, %Y") if patient and patient.created_at else None,
        }

        return {
            "patient": patient_dict,
            "current_screening": current_screening,
            "previous_screening": previous_screening,
            "current_score": current_score,
            "previous_score": previous_score,
            "diff_points": diff_points,
            "trend_direction": trend_direction,
            "trend_status": trend_status,
            "trend_icon": trend_icon,
            "trend_color": trend_color,
            "trend_badge_class": trend_badge_class,
            "trend_description": trend_description,
            "screenings": timeline_records,
            "total_screenings": len(all_screenings),
            "chart_data": {
                "labels": chart_labels,
                "scores": chart_scores,
                "predictions": chart_predictions,
                "screening_ids": chart_ids,
                "tooltips": chart_tooltips,
                "threshold": round(current_screening.threshold * 100.0, 1),
            },
        }
    except Exception as e:
        logger.exception("Error calculating patient progress data:")
        return None


def generate_patient_pdf_report(screening_id: int | None = None, patient_id: str | None = None) -> bytes | None:
    """
    Retrieves progress data and generates a professional PDF report.
    Returns the PDF file bytes.
    """
    from .report_generator import generate_patient_progress_pdf
    progress_data = get_patient_progress_data(screening_id=screening_id, patient_id=patient_id)
    if not progress_data:
        return None
    return generate_patient_progress_pdf(progress_data)

