"""Unit and integration tests for ANEMEIA database integration and user management."""
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
from src.database.models import db, User, ScreeningHistory
from src.database.service import (
    register_user,
    authenticate_user,
    save_screening,
    get_user_screenings,
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

    def test_user_registration_and_password_hashing(self):
        with app.app_context():
            # 1. Successful registration
            user, err = register_user("Dr. Alice Smith", "alice@hospital.org", "SecurePass123", "+1234567890")
            self.assertIsNotNone(user)
            self.assertIsNone(err)
            self.assertEqual(user.full_name, "Dr. Alice Smith")
            self.assertEqual(user.email, "alice@hospital.org")
            self.assertNotEqual(user.password_hash, "SecurePass123")
            self.assertTrue(user.check_password("SecurePass123"))
            self.assertFalse(user.check_password("WrongPassword"))

            # 2. Duplicate email rejection
            user2, err2 = register_user("Another Alice", "alice@hospital.org", "Pass456789")
            self.assertIsNone(user2)
            self.assertIn("already exists", err2)

            # 3. Invalid email rejection
            user3, err3 = register_user("Bob", "invalid-email-address", "Pass123456")
            self.assertIsNone(user3)
            self.assertIn("valid email", err3)

            # 4. Short password rejection
            user4, err4 = register_user("Charlie", "charlie@hospital.org", "123")
            self.assertIsNone(user4)
            self.assertIn("at least 6 characters", err4)

    def test_user_authentication(self):
        with app.app_context():
            register_user("Dr. Bob Wilson", "bob@hospital.org", "Password987")

            # Correct login
            user, err = authenticate_user("bob@hospital.org", "Password987")
            self.assertIsNotNone(user)
            self.assertIsNone(err)
            self.assertEqual(user.email, "bob@hospital.org")

            # Wrong password
            user_bad, err_bad = authenticate_user("bob@hospital.org", "WrongPassword")
            self.assertIsNone(user_bad)
            self.assertIn("Invalid email or password", err_bad)

            # Non-existent email
            user_none, err_none = authenticate_user("nobody@hospital.org", "Password987")
            self.assertIsNone(user_none)
            self.assertIn("Invalid email or password", err_none)

    def test_screening_persistence_and_isolation(self):
        with app.app_context():
            # Register two distinct users
            user_a, _ = register_user("User A", "user_a@hospital.org", "PassUserA")
            user_b, _ = register_user("User B", "user_b@hospital.org", "PassUserB")

            # Create synthetic test images
            img_bgr = np.full((224, 224, 3), 150, dtype=np.uint8)
            mask_uint8 = np.ones((224, 224), dtype=np.uint8)
            roi_bgr = np.full((224, 224, 3), 180, dtype=np.uint8)
            gradcam_bgr = np.full((224, 224, 3), 200, dtype=np.uint8)

            # Save screening for User A
            rec_a, err_a = save_screening(
                original_bgr=img_bgr,
                binary_mask=mask_uint8,
                roi_bgr=roi_bgr,
                gradcam_bgr=gradcam_bgr,
                prediction="Anemic",
                confidence_score=0.8245,
                threshold=0.3837,
                tissue_quality="accepted",
                user_id=user_a.id,
                patient_age=45,
                patient_gender="Female",
                upload_dir=app.config["UPLOAD_FOLDER"],
            )
            self.assertIsNotNone(rec_a)
            self.assertIsNone(err_a)
            self.assertEqual(rec_a.prediction, "Anemic")
            self.assertTrue(rec_a.is_anemic)
            self.assertEqual(rec_a.confidence_percentage, 82.5)

            # Verify files exist on disk
            orig_file = os.path.join(app.config["UPLOAD_FOLDER"], rec_a.original_image_path)
            self.assertTrue(os.path.exists(orig_file))

            # Test User Isolation: User A should see record, User B should NOT have access
            rec_found, err_found = get_screening_by_id(rec_a.id, user_id=user_a.id)
            self.assertIsNotNone(rec_found)
            self.assertIsNone(err_found)

            rec_denied, err_denied = get_screening_by_id(rec_a.id, user_id=user_b.id)
            self.assertIsNone(rec_found_denied := rec_denied)
            self.assertIn("Access denied", err_denied)

            # Test history pagination for User A
            p_a = get_user_screenings(user_id=user_a.id)
            self.assertEqual(len(p_a.items), 1)

            # History for User B should be empty
            p_b = get_user_screenings(user_id=user_b.id)
            self.assertEqual(len(p_b.items), 0)

    def test_flask_routes(self):
        # 1. GET / should return 200
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)

        # 2. GET /history should return 200
        res_h = self.client.get("/history")
        self.assertEqual(res_h.status_code, 200)

        # 3. GET /login should return 200
        res_l = self.client.get("/login")
        self.assertEqual(res_l.status_code, 200)

        # 4. GET /register should return 200
        res_r = self.client.get("/register")
        self.assertEqual(res_r.status_code, 200)

        # 5. POST /register through Flask client
        res_post_reg = self.client.post("/register", data={
            "full_name": "Dr. Test Clinical",
            "email": "clinical@example.com",
            "password": "Password12345",
            "phone_number": "+15551234",
        }, follow_redirects=True)
        self.assertEqual(res_post_reg.status_code, 200)

        # 6. GET /dashboard should return 200
        res_d = self.client.get("/dashboard")
        self.assertEqual(res_d.status_code, 200)


if __name__ == "__main__":
    unittest.main()
