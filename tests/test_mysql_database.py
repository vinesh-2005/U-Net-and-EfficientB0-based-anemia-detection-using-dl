"""Unit and integration tests for ANEMEIA patient screening history and database integration."""
import os
import shutil
import tempfile
import unittest
import numpy as np
import cv2

# Ensure tests use an isolated test database
os.environ["DB_FALLBACK_SQLITE"] = "true"
os.environ["SECRET_KEY"] = "test-secret-key-12345"

from app import app
from src.database.models import db, ScreeningHistory
from src.database.service import (
    save_screening,
    get_all_screenings,
    get_screening_by_id,
    check_db_connection,
)


class DatabaseIntegrationTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_anemeia.db")
        app.config["TESTING"] = True
        app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{self.db_path}"
        app.config["UPLOAD_FOLDER"] = os.path.join(self.temp_dir, "uploads")
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        self.client = app.test_client()

        with app.app_context():
            db.create_all()

    def tearDown(self):
        with app.app_context():
            db.session.remove()
            db.drop_all()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_database_connection_check(self):
        with app.app_context():
            is_connected, err = check_db_connection()
            self.assertTrue(is_connected)
            self.assertIsNone(err)

    def test_screening_persistence_with_patient_name_and_age(self):
        with app.app_context():
            # Create synthetic test images
            img_bgr = np.full((224, 224, 3), 150, dtype=np.uint8)
            mask_uint8 = np.ones((224, 224), dtype=np.uint8)
            roi_bgr = np.full((224, 224, 3), 180, dtype=np.uint8)
            gradcam_bgr = np.full((224, 224, 3), 200, dtype=np.uint8)

            # Save screening for Patient 1
            rec_1, err_1 = save_screening(
                original_bgr=img_bgr,
                binary_mask=mask_uint8,
                roi_bgr=roi_bgr,
                gradcam_bgr=gradcam_bgr,
                prediction="Anemic",
                confidence_score=0.8245,
                threshold=0.3837,
                tissue_quality="accepted",
                patient_name="Sarah Connor",
                patient_age=34,
                patient_gender="Female",
                upload_dir=app.config["UPLOAD_FOLDER"],
            )
            self.assertIsNotNone(rec_1)
            self.assertIsNone(err_1)
            self.assertEqual(rec_1.patient_name, "Sarah Connor")
            self.assertEqual(rec_1.patient_age, 34)
            self.assertEqual(rec_1.patient_gender, "Female")
            self.assertEqual(rec_1.prediction, "Anemic")
            self.assertTrue(rec_1.is_anemic)
            self.assertEqual(rec_1.confidence_percentage, 82.5)

            # Verify files exist on disk
            orig_file = os.path.join(app.config["UPLOAD_FOLDER"], rec_1.original_image_path)
            self.assertTrue(os.path.exists(orig_file))

            # Save screening for Patient 2 (anonymous)
            rec_2, err_2 = save_screening(
                original_bgr=img_bgr,
                binary_mask=mask_uint8,
                roi_bgr=roi_bgr,
                gradcam_bgr=gradcam_bgr,
                prediction="Non-Anemic (Normal)",
                confidence_score=0.1520,
                threshold=0.3837,
                tissue_quality="accepted",
                patient_name="",  # Should default to Anonymous Patient
                patient_age=42,
                patient_gender="Male",
                upload_dir=app.config["UPLOAD_FOLDER"],
            )
            self.assertIsNotNone(rec_2)
            self.assertEqual(rec_2.patient_name, "Anonymous Patient")
            self.assertEqual(rec_2.patient_age, 42)
            self.assertFalse(rec_2.is_anemic)

            # Retrieve by ID directly
            rec_found, err_found = get_screening_by_id(rec_1.id)
            self.assertIsNotNone(rec_found)
            self.assertIsNone(err_found)
            self.assertEqual(rec_found.patient_name, "Sarah Connor")

            # Test all screenings pagination
            pagination = get_all_screenings(page=1, per_page=10)
            self.assertIsNotNone(pagination)
            self.assertEqual(len(pagination.items), 2)

            # Test search by patient name
            p_search = get_all_screenings(page=1, per_page=10, search="Sarah")
            self.assertIsNotNone(p_search)
            self.assertEqual(len(p_search.items), 1)
            self.assertEqual(p_search.items[0].patient_name, "Sarah Connor")

    def test_flask_routes(self):
        # 1. GET / should return 200 and include patient_name field
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"patient_name", res.data)
        self.assertIn(b"Patient Full Name", res.data)

        # 2. GET /history should return 200 and show Patient History
        res_h = self.client.get("/history")
        self.assertEqual(res_h.status_code, 200)
        self.assertIn(b"Patient Screening History", res_h.data)

        # 3. GET /dashboard should return 200
        res_d = self.client.get("/dashboard")
        self.assertEqual(res_d.status_code, 200)


if __name__ == "__main__":
    unittest.main()
