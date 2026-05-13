import os

import requests


def get_telegram_bot_token():
    return (os.getenv("TELEGRAM_BOT_TOKEN") or "").strip()


def get_telegram_bot_username():
    return (os.getenv("TELEGRAM_BOT_USERNAME") or "").strip()


def telegram_bot_configured():
    return bool(get_telegram_bot_token())


def build_telegram_bot_link():
    username = get_telegram_bot_username()
    return f"https://t.me/{username}" if username else None


def send_telegram_message(chat_id, text, timeout=10):
    token = get_telegram_bot_token()
    if not token:
        return {
            "ok": False,
            "status": "bot_not_configured",
            "error": "Telegram bot token is not configured in the environment.",
        }

    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={
                "chat_id": chat_id,
                "text": text,
            },
            timeout=timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not payload.get("ok"):
            return {
                "ok": False,
                "status": "telegram_api_error",
                "error": str(payload),
            }
        result = payload.get("result", {})
        return {
            "ok": True,
            "status": "sent",
            "message_id": result.get("message_id"),
        }
    except requests.RequestException as error:
        return {
            "ok": False,
            "status": "request_failed",
            "error": str(error),
        }
