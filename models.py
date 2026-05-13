"""SQLAlchemy models for the user side of LiveStrat.

The analytics results live in CSVs, not in the database. The only
things stored in Postgres are the user, their alert preferences, their
saved strategy profiles, and the in-app notifications they have been
sent.
"""

from datetime import datetime

from werkzeug.security import check_password_hash, generate_password_hash

from database import db


class User(db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    full_name = db.Column(db.String(120), nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    last_login_at = db.Column(db.DateTime, nullable=True)

    alert_preferences = db.relationship(
        "AlertPreference",
        back_populates="user",
        uselist=False,
        cascade="all, delete-orphan",
    )
    saved_strategies = db.relationship(
        "SavedStrategyProfile",
        back_populates="user",
        cascade="all, delete-orphan",
    )
    notification_events = db.relationship(
        "NotificationEvent",
        back_populates="user",
        cascade="all, delete-orphan",
    )

    def set_password(self, password):
        # PBKDF2 with a per-user salt via werkzeug. Never stored plaintext.
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class AlertPreference(db.Model):
    __tablename__ = "alert_preferences"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, unique=True)
    price_alerts = db.Column(db.Boolean, nullable=False, default=True)
    strategy_alerts = db.Column(db.Boolean, nullable=False, default=True)
    sentiment_alerts = db.Column(db.Boolean, nullable=False, default=True)
    onchain_alerts = db.Column(db.Boolean, nullable=False, default=True)
    volatility_alerts = db.Column(db.Boolean, nullable=False, default=True)
    telegram_enabled = db.Column(db.Boolean, nullable=False, default=False)
    telegram_chat_id = db.Column(db.String(120), nullable=True)
    updated_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    user = db.relationship("User", back_populates="alert_preferences")

    def to_alert_engine_preferences(self):
        return {
            "price": self.price_alerts,
            "strategy": self.strategy_alerts,
            "sentiment": self.sentiment_alerts,
            "onchain": self.onchain_alerts,
            "volatility": self.volatility_alerts,
        }


class SavedStrategyProfile(db.Model):
    """A named strategy configuration that a logged-in user has saved.

    The configuration is kept as JSON so changes in the analytics layer
    do not need a database migration.
    """

    __tablename__ = "saved_strategy_profiles"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    name = db.Column(db.String(120), nullable=False)
    asset = db.Column(db.String(20), nullable=False, default="BTCUSDT")
    profile_tag = db.Column(db.String(80), nullable=True)
    config_json = db.Column(db.JSON, nullable=False, default=dict)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    updated_at = db.Column(
        db.DateTime,
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    user = db.relationship("User", back_populates="saved_strategies")


class NotificationEvent(db.Model):
    __tablename__ = "notification_events"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False, index=True)
    symbol = db.Column(db.String(20), nullable=False)
    category = db.Column(db.String(40), nullable=False)
    severity = db.Column(db.String(20), nullable=False)
    title = db.Column(db.String(160), nullable=False)
    message = db.Column(db.Text, nullable=False)
    action = db.Column(db.String(200), nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    read_at = db.Column(db.DateTime, nullable=True)

    user = db.relationship("User", back_populates="notification_events")
