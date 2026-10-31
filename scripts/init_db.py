#!/usr/bin/env python3
"""
Database initialization and connection verification utility for ANEMEIA.
Usage:
    python scripts/init_db.py
"""
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(BASE_DIR, ".env"))

from src.database.config import get_db_uri
from src.database.models import db, User, ScreeningHistory
from src.database.service import check_db_connection
from flask import Flask


def run_init():
    print("=" * 65)
    print("  ANEMEIA — MySQL Database Initialization & Verification")
    print("=" * 65)

    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "3306")
    name = os.environ.get("DB_NAME", "anemeia_db")
    user = os.environ.get("DB_USER", "root")

    print(f"\n[1/3] Target Configuration:")
    print(f"      Host:     {host}:{port}")
    print(f"      Database: {name}")
    print(f"      Username: {user}")

    app = Flask(__name__)
    app.config["SQLALCHEMY_DATABASE_URI"] = get_db_uri(BASE_DIR)
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    db.init_app(app)

    print("\n[2/3] Checking Database Server Connection...")
    with app.app_context():
        is_connected, err = check_db_connection()
        if not is_connected:
            print(f"\n[ERROR] Connection failed: {err}")
            print("\nTroubleshooting Steps:")
            print("  1. Verify MySQL service is started (e.g. Services -> MySQL, or XAMPP / MySQL Workbench).")
            print("  2. Ensure the database 'anemeia_db' exists:")
            print(f"     mysql -u {user} -p -e 'CREATE DATABASE IF NOT EXISTS {name};'")
            print("  3. Check credentials in your local .env file.")
            sys.exit(1)

        print("      [SUCCESS] MySQL server connected successfully!")

        print("\n[3/3] Creating/verifying database tables (users, screening_history)...")
        try:
            db.create_all()
            print("      [SUCCESS] Tables verified/created successfully!")
            print("        - users")
            print("        - screening_history")
        except Exception as e:
            print(f"      [ERROR] Could not create tables: {e}")
            sys.exit(1)

    print("\n" + "=" * 65)
    print("  DATABASE READY: ANEMEIA application is ready to save screenings!")
    print("=" * 65)


if __name__ == "__main__":
    run_init()
