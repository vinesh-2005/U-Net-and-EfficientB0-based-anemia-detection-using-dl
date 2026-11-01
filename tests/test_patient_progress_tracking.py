"""Unit and integration tests for Patient Anemia Progress Tracking, Graph Data, and PDF Report."""
import os
import shutil
import tempfile
import unittest
import numpy as np

os.environ["DB_FALLBACK_SQLITE"] = "true"
os.environ["SECRET_KEY"] = "test-secret-key-12345"

from app import app
from src.database.models import db, Patient, ScreeningHistory
from src.database.service import (
    save_screening,
    get_or_create_patient,
    get_patient_progress_data,
    generate_patient_pdf_report,
)


class PatientProgressTrackingTestCase(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.temp_dir, "test_progress.db")
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

    def test_repeat_patient_visits_and_progress_tracking(self):
        with app.app_context():
            dummy_img = np.full((224, 224, 3), 128, dtype=np.uint8)
            dummy_mask = np.ones((224, 224), dtype=np.uint8)
            dummy_roi = np.full((224, 224, 3), 160, dtype=np.uint8)

            # Visit 1 for Patient "Vinesh"
            rec_1, err_1 = save_screening(
                original_bgr=dummy_img,
                binary_mask=dummy_mask,
                roi_bgr=dummy_roi,
                gradcam_bgr=None,
                prediction="Anemic",
                confidence_score=0.650,  # 65.0%
                threshold=0.3837,
                tissue_quality="accepted",
                patient_name="Vinesh",
                patient_age=21,
                patient_gender="Male",
                upload_dir=app.config["UPLOAD_FOLDER"],
            )
            self.assertIsNotNone(rec_1)
            self.assertIsNone(err_1)
            assigned_patient_id = rec_1.patient_id
            self.assertTrue(assigned_patient_id.startswith("PID-"))

            # Visit 1 Progress Check: Should establish baseline
            prog_1 = get_patient_progress_data(screening_id=rec_1.id)
            self.assertIsNotNone(prog_1)
            self.assertEqual(prog_1["trend_status"], "Initial Baseline Established")
            self.assertIsNone(prog_1["previous_score"])
            self.assertEqual(prog_1["current_score"], 65.0)
            self.assertIsNone(prog_1["diff_points"])
            self.assertEqual(len(prog_1["screenings"]), 1)

            # Visit 2 for same Patient "Vinesh" (decreased score -> 52%)
            rec_2, err_2 = save_screening(
                original_bgr=dummy_img,
                binary_mask=dummy_mask,
                roi_bgr=dummy_roi,
                gradcam_bgr=None,
                prediction="Anemic",
                confidence_score=0.520,  # 52.0%
                threshold=0.3837,
                tissue_quality="accepted",
                patient_name="Vinesh",  # Same name -> must reuse same Patient ID
                patient_age=21,
                patient_gender="Male",
                upload_dir=app.config["UPLOAD_FOLDER"],
            )
            self.assertIsNotNone(rec_2)
            self.assertEqual(rec_2.patient_id, assigned_patient_id)

            # Visit 2 Progress Check: Trend Decreased
            prog_2 = get_patient_progress_data(screening_id=rec_2.id)
            self.assertIsNotNone(prog_2)
            self.assertEqual(prog_2["previous_score"], 65.0)
            self.assertEqual(prog_2["current_score"], 52.0)
            self.assertEqual(prog_2["diff_points"], -13.0)
            self.assertEqual(prog_2["trend_direction"], "decreased")
            self.assertEqual(prog_2["trend_status"], "Risk Score Decreased")
            self.assertEqual(prog_2["trend_icon"], "↓")
            self.assertEqual(len(prog_2["screenings"]), 2)
            self.assertEqual(len(prog_2["chart_data"]["labels"]), 2)
            self.assertEqual(prog_2["chart_data"]["scores"], [65.0, 52.0])

            # Visit 3 for same Patient "Vinesh" (further decrease -> 44%)
            rec_3, err_3 = save_screening(
                original_bgr=dummy_img,
                binary_mask=dummy_mask,
                roi_bgr=dummy_roi,
                gradcam_bgr=None,
                prediction="Anemic",
                confidence_score=0.440,  # 44.0%
                threshold=0.3837,
                tissue_quality="accepted",
                patient_id=assigned_patient_id,  # Explicit ID lookup
                patient_name="Vinesh",
                patient_age=21,
                patient_gender="Male",
                upload_dir=app.config["UPLOAD_FOLDER"],
            )
            self.assertIsNotNone(rec_3)
            self.assertEqual(rec_3.patient_id, assigned_patient_id)

            prog_3 = get_patient_progress_data(screening_id=rec_3.id)
            self.assertEqual(prog_3["previous_score"], 52.0)
            self.assertEqual(prog_3["current_score"], 44.0)
            self.assertEqual(prog_3["diff_points"], -8.0)
            self.assertEqual(prog_3["trend_status"], "Risk Score Decreased")
            self.assertEqual(len(prog_3["screenings"]), 3)

            # Visit 4: Risk score increased test
            rec_4, _ = save_screening(
                original_bgr=dummy_img,
                binary_mask=dummy_mask,
                roi_bgr=dummy_roi,
                gradcam_bgr=None,
                prediction="Anemic",
                confidence_score=0.600,  # 60.0%
                threshold=0.3837,
                patient_id=assigned_patient_id,
                upload_dir=app.config["UPLOAD_FOLDER"],
            )
            prog_4 = get_patient_progress_data(screening_id=rec_4.id)
            self.assertEqual(prog_4["diff_points"], 16.0)
            self.assertEqual(prog_4["trend_status"], "Risk Score Increased")
            self.assertEqual(prog_4["trend_icon"], "↑")

    def test_pdf_report_generation(self):
        with app.app_context():
            dummy_img = np.full((224, 224, 3), 128, dtype=np.uint8)
            dummy_mask = np.ones((224, 224), dtype=np.uint8)
            dummy_roi = np.full((224, 224, 3), 160, dtype=np.uint8)

            rec, _ = save_screening(
                original_bgr=dummy_img,
                binary_mask=dummy_mask,
                roi_bgr=dummy_roi,
                gradcam_bgr=None,
                prediction="Non-Anemic (Normal)",
                confidence_score=0.225,
                threshold=0.3837,
                patient_name="Eleanor Vance",
                patient_age=28,
                patient_gender="Female",
                upload_dir=app.config["UPLOAD_FOLDER"],
            )

            pdf_bytes = generate_patient_pdf_report(screening_id=rec.id)
            self.assertIsNotNone(pdf_bytes)
            self.assertTrue(len(pdf_bytes) > 1000)
            # Confirm PDF magic header bytes %PDF-
            self.assertTrue(pdf_bytes.startswith(b"%PDF"))

    def test_flask_progress_and_pdf_routes(self):
        with app.app_context():
            dummy_img = np.full((224, 224, 3), 128, dtype=np.uint8)
            dummy_mask = np.ones((224, 224), dtype=np.uint8)
            dummy_roi = np.full((224, 224, 3), 160, dtype=np.uint8)

            rec, _ = save_screening(
                original_bgr=dummy_img,
                binary_mask=dummy_mask,
                roi_bgr=dummy_roi,
                gradcam_bgr=None,
                prediction="Anemic",
                confidence_score=0.784,
                threshold=0.3837,
                patient_name="Carlos Mendez",
                patient_age=45,
                patient_gender="Male",
                upload_dir=app.config["UPLOAD_FOLDER"],
            )
            screening_id = rec.id
            patient_id = rec.patient_id

        # 1. GET /history/<id> should return 200 and include Progress Tracking elements
        res = self.client.get(f"/history/{screening_id}")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"Patient Anemia Progress Tracking", res.data)
        self.assertIn(b"progressChart", res.data)
        self.assertIn(b"Download Patient Progress Report (PDF)", res.data)
        self.assertIn(b"Initial Baseline Established", res.data)

        # 2. GET /history/<id>/report.pdf should return 200 with application/pdf
        res_pdf = self.client.get(f"/history/{screening_id}/report.pdf")
        self.assertEqual(res_pdf.status_code, 200)
        self.assertEqual(res_pdf.content_type, "application/pdf")
        self.assertTrue(res_pdf.data.startswith(b"%PDF"))

        # 3. GET /patient/<patient_id>/report.pdf should return 200 with application/pdf
        res_pat_pdf = self.client.get(f"/patient/{patient_id}/report.pdf")
        self.assertEqual(res_pat_pdf.status_code, 200)
        self.assertEqual(res_pat_pdf.content_type, "application/pdf")
        self.assertTrue(res_pat_pdf.data.startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
