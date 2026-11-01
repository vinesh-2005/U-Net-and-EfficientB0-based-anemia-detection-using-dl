"""ANEMEIA Database package."""
from .models import db, User, Patient, ScreeningHistory
from .service import (
    check_db_connection,
    init_db,
    register_user,
    authenticate_user,
    get_or_create_patient,
    save_screening,
    get_all_screenings,
    get_user_screenings,
    get_screening_by_id,
    get_patient_progress_data,
    generate_patient_pdf_report,
)
from .config import get_db_uri

__all__ = [
    "db",
    "User",
    "Patient",
    "ScreeningHistory",
    "check_db_connection",
    "init_db",
    "register_user",
    "authenticate_user",
    "get_or_create_patient",
    "save_screening",
    "get_all_screenings",
    "get_user_screenings",
    "get_screening_by_id",
    "get_patient_progress_data",
    "generate_patient_pdf_report",
    "get_db_uri",
]
