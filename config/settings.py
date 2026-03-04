import os
from pathlib import Path
from dataclasses import dataclass, field


def _load_dotenv(env_path: Path):
    """Load .env file into os.environ (no third-party dependency)."""
    if not env_path.exists():
        return
    with open(env_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if value and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            os.environ.setdefault(key, value)


# Load .env before Settings is instantiated
_load_dotenv(Path(__file__).resolve().parent.parent / ".env")


@dataclass
class Settings:
    # Paths
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
    DB_PATH: Path = field(default=None)
    MODELS_DIR: Path = field(default=None)
    LOGS_DIR: Path = field(default=None)
    SIGNALS_DIR: Path = field(default=None)

    # Email (Gmail SMTP)
    EMAIL_SENDER: str = field(default_factory=lambda: os.environ.get("EMAIL_SENDER", ""))
    EMAIL_PASSWORD: str = field(default_factory=lambda: os.environ.get("EMAIL_PASSWORD", ""))
    EMAIL_RECIPIENT: str = field(default_factory=lambda: os.environ.get("EMAIL_RECIPIENT", ""))
    SMTP_SERVER: str = "smtp.gmail.com"
    SMTP_PORT: int = 587

    # Data settings
    BACKFILL_START: str = "2023-01-01"
    OHLCV_LOOKBACK_DAYS: int = 500
    FEATURE_LOOKBACK_DAYS: int = 60

    # Model settings
    TARGET_THRESHOLD: float = 0.01  # 1% movement threshold
    MIN_TRAIN_DAYS: int = 252
    VAL_WINDOW_DAYS: int = 21
    RETRAIN_FREQUENCY: str = "weekly"

    # Signal settings
    CONFIDENCE_THRESHOLD: float = 0.45
    STRONG_CONFIDENCE: float = 0.55
    MIN_EDGE: float = 0.10

    # Risk settings
    MAX_POSITIONS: int = 10
    MAX_PER_STOCK_PCT: float = 0.10
    MAX_SECTOR_PCT: float = 0.30
    MAX_LONG_PCT: float = 0.60
    MAX_SHORT_PCT: float = 0.40
    MAX_TOTAL_EXPOSURE_PCT: float = 0.80

    # API settings
    NEWS_FETCH_DELAY_SECS: float = 1.5
    YFINANCE_BATCH_SIZE: int = 50
    MAX_RETRIES: int = 3
    REQUEST_TIMEOUT_SECS: int = 30

    # LightGBM parameters
    LGB_PARAMS: dict = field(default_factory=lambda: {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_estimators": 500,
        "class_weight": "balanced",
        "min_child_samples": 20,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
    })

    # XGBoost parameters
    XGB_PARAMS: dict = field(default_factory=lambda: {
        "objective": "multi:softprob",
        "num_class": 3,
        "eval_metric": "mlogloss",
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 500,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "min_child_weight": 5,
        "tree_method": "hist",
        "verbosity": 0,
    })

    # Random Forest parameters
    RF_PARAMS: dict = field(default_factory=lambda: {
        "n_estimators": 300,
        "max_depth": 12,
        "min_samples_leaf": 10,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": 42,
    })

    def __post_init__(self):
        if self.DB_PATH is None:
            self.DB_PATH = self.PROJECT_ROOT / "storage" / "nifty_predictor.db"
        if self.MODELS_DIR is None:
            self.MODELS_DIR = self.PROJECT_ROOT / "storage" / "models"
        if self.LOGS_DIR is None:
            self.LOGS_DIR = self.PROJECT_ROOT / "storage" / "logs"
        if self.SIGNALS_DIR is None:
            self.SIGNALS_DIR = self.PROJECT_ROOT / "storage" / "signals"

        # Ensure directories exist
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.SIGNALS_DIR.mkdir(parents=True, exist_ok=True)


SETTINGS = Settings()
