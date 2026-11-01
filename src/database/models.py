"""SQLAlchemy models for ANEMEIA user management and screening history."""
from datetime import datetime, timezone
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


def utc_now():
    return datetime.now(timezone.utc)


class User(db.Model):
    """Registered user entity (clinicians, researchers, patients)."""
    __tablename__ = "users"

    id = db.Column(db.BigInteger().with_variant(db.Integer, "sqlite"), primary_key=True, autoincrement=True)
    full_name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(191), nullable=False, unique=True, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    phone_number = db.Column(db.String(20), nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=utc_now)

    # One-to-many relationship with screening history records
    screenings = db.relationship(
        "ScreeningHistory",
        backref="user",
        lazy="dynamic",
        cascade="all, delete-orphan",
    )

    def set_password(self, password: str):
        """Hashes the password securely using Werkzeug's modern scrypt/pbkdf2."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        """Validates a candidate password against the stored cryptographic hash."""
        return check_password_hash(self.password_hash, password)

    def to_dict(self):
        return {
            "id": self.id,
            "full_name": self.full_name,
            "email": self.email,
            "phone_number": self.phone_number,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else None,
        }


class Patient(db.Model):
    """Patient entity representing a tracked individual over multiple longitudinal screenings."""
    __tablename__ = "patients"

    id = db.Column(db.String(50), primary_key=True)  # e.g., "PID-1001"
    name = db.Column(db.String(120), nullable=False, index=True)
    age = db.Column(db.Integer, nullable=True)
    gender = db.Column(db.String(20), nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=utc_now)
    updated_at = db.Column(db.DateTime, nullable=False, default=utc_now, onupdate=utc_now)

    screenings = db.relationship(
        "ScreeningHistory",
        backref=db.backref("patient_profile", lazy="joined"),
        lazy="select",
        cascade="all, delete-orphan",
        order_by="ScreeningHistory.created_at.asc()",
    )

    @property
    def screening_count(self) -> int:
        return len(self.screenings)

    @property
    def latest_screening(self):
        return self.screenings[-1] if self.screenings else None

    @property
    def previous_screening(self):
        return self.screenings[-2] if len(self.screenings) >= 2 else None

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "age": self.age,
            "gender": self.gender,
            "screening_count": self.screening_count,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else None,
            "updated_at": self.updated_at.strftime("%Y-%m-%d %H:%M:%S") if self.updated_at else None,
        }


class ScreeningHistory(db.Model):
    """Historical record of an AI conjunctival screening assessment."""
    __tablename__ = "screening_history"

    id = db.Column(db.BigInteger().with_variant(db.Integer, "sqlite"), primary_key=True, autoincrement=True)
    patient_id = db.Column(db.String(50), db.ForeignKey("patients.id", ondelete="SET NULL"), nullable=True, index=True)
    patient_name = db.Column(db.String(120), nullable=True, default="Anonymous Patient", index=True)
    patient_age = db.Column(db.Integer, nullable=True)
    patient_gender = db.Column(db.String(20), nullable=True)
    user_id = db.Column(db.BigInteger().with_variant(db.Integer, "sqlite"), db.ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    original_image_path = db.Column(db.String(255), nullable=False)
    segmented_image_path = db.Column(db.String(255), nullable=False)
    extracted_roi_path = db.Column(db.String(255), nullable=False)
    gradcam_image_path = db.Column(db.String(255), nullable=True)
    prediction = db.Column(db.String(50), nullable=False)  # "Anemic" or "Non-Anemic"
    confidence_score = db.Column(db.Float, nullable=False)  # Raw classifier output (0.0 to 1.0)
    threshold = db.Column(db.Float, nullable=False, default=0.3837)
    model_name = db.Column(db.String(100), nullable=False, default="EfficientNet-B0 + U-Net")
    tissue_quality = db.Column(db.String(50), nullable=True, default="accepted")
    created_at = db.Column(db.DateTime, nullable=False, default=utc_now, index=True)

    @property
    def display_patient_id(self) -> str:
        return self.patient_id or (f"PID-{1000 + self.id}" if self.id else "PID-UNKNOWN")

    @property
    def is_anemic(self) -> bool:
        return self.prediction.lower().startswith("anemic") or self.confidence_score >= self.threshold

    @property
    def confidence_percentage(self) -> float:
        return round(self.confidence_score * 100.0, 1)

    @property
    def threshold_percentage(self) -> float:
        return round(self.threshold * 100.0, 1)

    def to_dict(self):
        return {
            "id": self.id,
            "patient_id": self.display_patient_id,
            "patient_name": self.patient_name or "Anonymous Patient",
            "patient_age": self.patient_age,
            "patient_gender": self.patient_gender,
            "original_image_path": self.original_image_path,
            "segmented_image_path": self.segmented_image_path,
            "extracted_roi_path": self.extracted_roi_path,
            "gradcam_image_path": self.gradcam_image_path,
            "prediction": self.prediction,
            "confidence_score": self.confidence_score,
            "confidence_percentage": self.confidence_percentage,
            "threshold": self.threshold,
            "threshold_percentage": self.threshold_percentage,
            "model_name": self.model_name,
            "tissue_quality": self.tissue_quality,
            "created_at": self.created_at.strftime("%Y-%m-%d %H:%M:%S") if self.created_at else None,
        }
