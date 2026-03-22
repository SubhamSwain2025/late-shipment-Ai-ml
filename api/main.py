from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

import io
import joblib
import boto3
from fastapi import FastAPI
from routers import landing, ping as ping_router, predict_late, predict_very_late
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("app")

# Environment mode
APP_ENV = os.getenv("APP_ENV", "prod")
IS_TEST = (APP_ENV == "test")

# Create FastAPI app
app = FastAPI(
    title="Shipment Delay Prediction API",
    description="FastAPI — Local/Test + AWS (S3) support",
    version="1.0.0",
)

# Load environment variables
BUCKET = os.environ.get("ARTIFACT_BUCKET")
SCALER_KEY = os.environ.get("SCALER_KEY")
ONEHOT_KEY = os.environ.get("ONEHOT_KEY")
ORDINAL_KEY = os.environ.get("ORDINAL_KEY")
LATE_KEY = os.environ.get("LATE_MODEL_KEY")
VERY_LATE_KEY = os.environ.get("VERY_LATE_MODEL_KEY")

# Validate env variables (only in prod mode)
if not IS_TEST:
    required_vars = {
        "ARTIFACT_BUCKET": BUCKET,
        "SCALER_KEY": SCALER_KEY,
        "ONEHOT_KEY": ONEHOT_KEY,
        "ORDINAL_KEY": ORDINAL_KEY,
        "LATE_MODEL_KEY": LATE_KEY,
        "VERY_LATE_MODEL_KEY": VERY_LATE_KEY,
    }

    missing = [name for name, value in required_vars.items() if not value]

    if missing:
        raise RuntimeError("Missing environment variables: " + ", ".join(missing))


# S3 client (only if not test)
s3 = None if IS_TEST else boto3.client("s3")


# Load from S3
def load_joblib_from_s3(bucket: str, key: str):
    buf = io.BytesIO()
    s3.download_fileobj(bucket, key, buf)
    buf.seek(0)
    return joblib.load(buf)


# Startup event
@app.on_event("startup")
def load_artifacts():
    if IS_TEST:
        log.info("APP_ENV=test: loading LOCAL models...")

        try:
            app.state.scaler = joblib.load(SCALER_KEY)
            app.state.onehot = joblib.load(ONEHOT_KEY)
            app.state.ordinal = joblib.load(ORDINAL_KEY)
            app.state.late_model = joblib.load(LATE_KEY)
            app.state.very_late_model = joblib.load(VERY_LATE_KEY)

            log.info("✅ Local models loaded successfully")

        except Exception as e:
            log.error(f"❌ Failed to load local models: {e}")
            raise

    else:
        log.info("APP_ENV=prod: loading from S3...")

        try:
            app.state.scaler = load_joblib_from_s3(BUCKET, SCALER_KEY)
            app.state.onehot = load_joblib_from_s3(BUCKET, ONEHOT_KEY)
            app.state.ordinal = load_joblib_from_s3(BUCKET, ORDINAL_KEY)
            app.state.late_model = load_joblib_from_s3(BUCKET, LATE_KEY)
            app.state.very_late_model = load_joblib_from_s3(BUCKET, VERY_LATE_KEY)

            log.info("✅ S3 models loaded successfully")

        except Exception as e:
            log.error(f"❌ Failed to load S3 models: {e}")
            raise


# Routers
app.include_router(landing.router)
app.include_router(ping_router.router)
app.include_router(predict_late.router)
app.include_router(predict_very_late.router)