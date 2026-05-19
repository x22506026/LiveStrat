"""Account-side helpers used by the Flask routes.

Looking up the current user, building the account and notification
payloads, and pushing optional Telegram messages all live here so the
route functions in app.py stay short and easy to read.
"""

from flask import session

from database import db
from models import AlertPreference, NotificationEvent, User
from telegram_delivery import (
    build_telegram_bot_link,
    get_telegram_bot_username,
    send_telegram_message,
    telegram_bot_configured,
)


DEMO_DEFAULT_NAME = "LiveStrat User"


# Small helpers that strip whitespace and lowercase user input before it hits the database, so logins and lookups stay case-insensitive.
def normalize_email(email):
    return (email or "").strip().lower()


def normalize_username(username):
    return (username or "").strip().lower()


# Pulls the currently logged-in user out of the Flask session, or returns None if nobody is signed in. The rest of the routes lean on this.
def get_current_user():
    user_id = session.get("user_id")
    if not user_id:
        return None
    return db.session.get(User, user_id)


def get_demo_user():
    current_user = get_current_user()
    if not current_user:
        return None
    return current_user.username or current_user.full_name or DEMO_DEFAULT_NAME


def get_or_create_alert_preferences(user):
    if not user.alert_preferences:
        user.alert_preferences = AlertPreference()
        db.session.commit()
    return user.alert_preferences


def alert_preferences_payload(preferences):
    return {
        "price": preferences.price_alerts,
        "strategy": preferences.strategy_alerts,
        "sentiment": preferences.sentiment_alerts,
        "onchain": preferences.onchain_alerts,
        "volatility": preferences.volatility_alerts,
        "telegram_enabled": preferences.telegram_enabled,
        "telegram_chat_id": preferences.telegram_chat_id or "",
    }


def telegram_status_payload(preferences=None):
    preferences = preferences or {}
    telegram_enabled = (
        preferences.telegram_enabled
        if hasattr(preferences, "telegram_enabled")
        else bool(preferences.get("telegram_enabled"))
    )
    telegram_chat_id = (
        preferences.telegram_chat_id
        if hasattr(preferences, "telegram_chat_id")
        else preferences.get("telegram_chat_id", "")
    )
    return {
        "bot_configured": telegram_bot_configured(),
        "bot_username": get_telegram_bot_username() or None,
        "bot_link": build_telegram_bot_link(),
        "user_telegram_enabled": bool(telegram_enabled),
        "user_chat_id_present": bool((telegram_chat_id or "").strip()),
    }


# Turns SQLAlchemy rows into plain dicts so the JSON API endpoints can send them straight to the frontend without any extra shaping.
def serialize_strategy_profile(profile):
    config = profile.config_json or {}
    timeframe_policy = config.get("timeframe_policy") or {}
    governance = config.get("governance") or {}
    return {
        "id": str(profile.id),
        "name": profile.name,
        "asset": profile.asset,
        "tag": profile.profile_tag or "research",
        "saved_at": profile.created_at.isoformat(),
        "updated_at": profile.updated_at.isoformat(),
        "config": config,
        "timeframe_scope": timeframe_policy.get("timeframe_scope", "single_timeframe"),
        "selected_timeframes": timeframe_policy.get("selected_timeframes", config.get("selection", {}).get("timeframes", [])),
        "resolved_timeframe": timeframe_policy.get("resolved_timeframe", config.get("resolved_timeframe")),
        "operational_family": (config.get("family_governance") or {}).get("operational_family"),
        "daily_confirmation_fit": config.get("daily_confirmation_fit"),
        "daily_posture_label": governance.get("daily_posture_label", config.get("daily_posture_label")),
        "daily_structural_label": governance.get("daily_structural_label", config.get("daily_structural_label")),
        "daily_structural_confidence": governance.get("daily_structural_confidence", config.get("daily_structural_confidence")),
        "deployment_tier": governance.get("deployment_tier"),
        "readiness_label": governance.get("readiness_label"),
    }


def serialize_notification_event(event):
    return {
        "id": str(event.id),
        "symbol": event.symbol,
        "category": event.category,
        "severity": event.severity,
        "title": event.title,
        "message": event.message,
        "action": event.action,
        "created_at": event.created_at.isoformat(),
        "read_at": event.read_at.isoformat() if event.read_at else None,
    }


def build_telegram_ready_payload(preferences, event):
    if not preferences.telegram_enabled or not preferences.telegram_chat_id:
        return None

    from datetime import datetime, timezone

    timestamp = datetime.now(timezone.utc).strftime("%H:%M UTC")
    headline = (event.title or "").strip()
    body = (event.message or "").strip()
    action = (event.action or "Open LiveStrat to review.").strip()
    severity = (event.severity or "").lower()
    severity_tag = f"[{severity.upper()}] " if severity in ("medium", "high") else ""

    lines = [
        f"LiveStrat | {event.symbol}",
        f"{severity_tag}{headline}",
        "",
        body,
        "",
        f"Action: {action}",
        f"({timestamp})",
    ]
    return {
        "chat_id": preferences.telegram_chat_id,
        "text": "\n".join(lines),
        "parse_mode": None,
    }


# Saves new alerts as NotificationEvent rows and, when the user opted in, also queues a Telegram message for the next dispatch call.
def persist_alert_events_for_user(user, alerts):
    preferences = get_or_create_alert_preferences(user)
    created_events = []

    for alert in alerts:
        event = NotificationEvent(
            user_id=user.id,
            symbol=(alert.get("symbol") or "BTCUSDT")[:20],
            category=(alert.get("category") or "general")[:40],
            severity=(alert.get("severity") or "low")[:20],
            title=(alert.get("title") or "LiveStrat alert")[:160],
            message=alert.get("message") or "",
            action=(alert.get("action") or "Review in LiveStrat.")[:200],
        )
        db.session.add(event)
        created_events.append(event)

    db.session.commit()

    return {
        "events": [serialize_notification_event(event) for event in created_events],
        "telegram_ready_messages": [
            payload
            for payload in (
                build_telegram_ready_payload(preferences, event)
                for event in created_events
            )
            if payload is not None
        ],
    }


def dispatch_telegram_ready_messages(persisted_events):
    attempts = []
    telegram_messages = persisted_events.get("telegram_ready_messages", [])

    if not telegram_messages:
        return attempts

    for index, telegram_payload in enumerate(telegram_messages):
        delivery_result = send_telegram_message(
            telegram_payload.get("chat_id"),
            telegram_payload.get("text"),
        )
        event_reference = None
        if index < len(persisted_events.get("events", [])):
            event_reference = persisted_events["events"][index]["id"]
        attempts.append(
            {
                "event_id": event_reference,
                "chat_id": telegram_payload.get("chat_id"),
                "status": delivery_result.get("status"),
                "ok": delivery_result.get("ok", False),
                "message_id": delivery_result.get("message_id"),
                "error": delivery_result.get("error"),
            }
        )

    return attempts
