"""Database service functions for authentication, screening records, and safe disk storage."""
import logging
import os
import re
import uuid
import cv2
import numpy as np
from sqlalchemy import text
from .models import db, User, ScreeningHistory

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


def init_db(app, base_dir=None):
    """
    Initializes Flask-SQLAlchemy with the application and attempts safe table creation.
    If database server is offline, catches the error so the web app can still start.
    """
    from .config import get_db_uri
    db_uri = get_db_uri(base_dir)

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
            logger.info("Database tables verified/created successfully.")
            return True, "Connected"
        except Exception as e:
            logger.warning(f"Could not initialize database tables: {e}")
            return False, str(e)


# --------------------------------------------------
# User Management & Authentication
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
    patient_name: str | None = None,
    patient_age: int | None = None,
    patient_gender: str | None = None,
    user_id: int | None = None,
    model_name: str = "EfficientNet-B0 + U-Net",
    upload_dir: str = None,
) -> tuple[ScreeningHistory | None, str | None]:
    """
    Persists screening images safely to disk and records metadata in the MySQL database.
    Returns (ScreeningHistory, None) on success, or (None, error_message) on failure.
    Never fabricates fake IDs or records.
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
            patient_name=clean_name,
            patient_age=patient_age if patient_age else None,
            patient_gender=patient_gender.capitalize() if patient_gender else None,
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
    Supports optional case-insensitive search by patient name.
    """
    try:
        query = ScreeningHistory.query
        if search:
            query = query.filter(ScreeningHistory.patient_name.ilike(f"%{search.strip()}%"))

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
