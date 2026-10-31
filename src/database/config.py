"""Database configuration and URI builder for ANEMEIA."""
import os
import urllib.parse
from dotenv import load_dotenv


def get_db_uri(base_dir=None):
    """
    Builds the database URI from environment variables or .env file.
    Supports MySQL (production/development) and optional SQLite fallback.
    """
    if base_dir:
        dotenv_path = os.path.join(base_dir, ".env")
        if os.path.exists(dotenv_path):
            load_dotenv(dotenv_path)
    else:
        load_dotenv()

    # Explicit DATABASE_URL override if provided
    custom_url = os.environ.get("DATABASE_URL")
    if custom_url:
        return custom_url

    # Check for optional SQLite fallback mode (e.g. for testing or offline mode)
    use_sqlite = os.environ.get("DB_FALLBACK_SQLITE", "false").strip().lower() in ("true", "1", "yes")
    if use_sqlite:
        sqlite_path = os.path.join(base_dir or ".", "anemeia_db.sqlite3")
        return f"sqlite:///{sqlite_path}"

    host = os.environ.get("DB_HOST", "localhost")
    port = os.environ.get("DB_PORT", "3306")
    name = os.environ.get("DB_NAME", "anemeia_db")
    user = os.environ.get("DB_USER", "root")
    password = os.environ.get("DB_PASSWORD", "")

    # URL-encode user and password to safely handle special characters
    user_encoded = urllib.parse.quote_plus(user)
    password_encoded = urllib.parse.quote_plus(password)

    return f"mysql+pymysql://{user_encoded}:{password_encoded}@{host}:{port}/{name}?charset=utf8mb4"
