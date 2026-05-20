from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AlertPreference:
    price: bool = True
    strategy: bool = True
    sentiment: bool = False
    onchain: bool = False
    volatility: bool = True


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _text(value: Any, default: str = "unavailable") -> str:
    if value in (None, ""):
        return default
    return str(value)


def _display_signal(signal: Any) -> str:
    return _text(signal, "unknown").replace("_", " ").title()


def _display_label(value: Any, default: str = "none") -> str:
    return _text(value, default).replace("_", " ")


def _severity_for_return(return_24h: float) -> str:
    if abs(return_24h) >= 5:
        return "high"
    if abs(return_24h) >= 2:
        return "medium"
    return "low"


def _make_alert(
    *,
    alert_id: str,
    category: str,
    severity: str,
    title: str,
    message: str,
    action: str,
    symbol: str,
) -> dict[str, str]:
    return {
        "id": alert_id,
        "symbol": symbol,
        "category": category,
        "severity": severity,
        "title": title,
        "message": message,
        "action": action,
    }


def parse_alert_preferences(raw_preferences: dict[str, Any] | None) -> AlertPreference:
    raw_preferences = raw_preferences or {}
    return AlertPreference(
        price=bool(raw_preferences.get("price", True)),
        strategy=bool(raw_preferences.get("strategy", True)),
        sentiment=bool(raw_preferences.get("sentiment", False)),
        onchain=bool(raw_preferences.get("onchain", False)),
        volatility=bool(raw_preferences.get("volatility", True)),
    )


def build_alerts_for_market_summary(
    symbol: str,
    summary: dict[str, Any],
    preferences: AlertPreference | None = None,
) -> list[dict[str, str]]:
    preferences = preferences or AlertPreference()
    alerts: list[dict[str, str]] = []

    latest_close = _num(summary.get("latest_close"))
    return_24h = _num(summary.get("latest_return_24h_pct"))
    volatility_label = _text(summary.get("volatility_status"), "unknown").lower()
    primary_signal = _text(
        summary.get("final_strategy_signal")
        or summary.get("selected_primary_signal")
        or summary.get("latest_signal")
        or summary.get("scaled_model_signal"),
        "unknown",
    )
    confidence = _num(
        abs(_num(summary.get("final_strategy_score"), 0.0))
        or summary.get("selected_primary_confidence")
        or summary.get("latest_signal_confidence")
        or summary.get("scaled_model_confidence")
    )
    predicted_return = _num(summary.get("predicted_return"))
    predicted_price = _num(summary.get("predicted_price"))
    indicator_signal = _text(summary.get("indicator_signal"), "unknown")
    policy_action = _text(summary.get("latest_action"), "not available")
    strategy_display_name = _text(summary.get("strategy_display_name"), "Balanced Default")
    strategy_timeframe = _text(summary.get("strategy_timeframe"), "4h")
    sentiment_source = _text(summary.get("latest_effective_sentiment_source"))
    sentiment_label = _text(summary.get("latest_effective_sentiment_label"))
    gdelt_label = _text(summary.get("latest_gdelt_regime_label"))
    gdelt_theme = _text(
        summary.get("latest_gdelt_dominant_event_theme") or summary.get("gdelt_dominant_event_theme"),
        "none",
    )
    gdelt_risk_theme = _text(
        summary.get("latest_gdelt_risk_event_theme") or summary.get("gdelt_risk_event_theme"),
        "none",
    )
    gdelt_supportive_theme = _text(
        summary.get("latest_gdelt_supportive_event_theme") or summary.get("gdelt_supportive_event_theme"),
        "none",
    )
    onchain_label = _text(summary.get("latest_onchain_regime_label"))
    onchain_support_driver = _text(summary.get("latest_onchain_primary_support_driver"), "none")
    onchain_risk_driver = _text(summary.get("latest_onchain_primary_risk_driver"), "none")
    onchain_participation_breadth = _num(summary.get("latest_onchain_participation_breadth_score"))
    onchain_structural_fragility = _num(summary.get("latest_onchain_structural_fragility_score"))
    onchain_status = _text(summary.get("latest_onchain_snapshot_status"))
    context_variant = _text(summary.get("multimodal_selected_context_variant"), "")

    if preferences.price:
        direction = "up" if return_24h >= 0 else "down"
        alerts.append(
            _make_alert(
                alert_id=f"{symbol}-price-move",
                symbol=symbol,
                category="price",
                severity=_severity_for_return(return_24h),
                title=f"{symbol} price moved {direction}",
                message=(
                    f"{symbol} is trading at {latest_close:,.2f} with a "
                    f"{return_24h:.2f}% move over 24 hours."
                ),
                action="Review market chart and price context.",
            )
        )

    if preferences.volatility and volatility_label not in {"unknown", "n/a", "low"}:
        alerts.append(
            _make_alert(
                alert_id=f"{symbol}-volatility",
                symbol=symbol,
                category="risk",
                severity="medium" if volatility_label == "medium" else "high",
                title=f"{symbol} volatility is {volatility_label}",
                message=(
                    f"The latest volatility state is {volatility_label}. "
                    "Position size and entry timing should be reviewed before acting."
                ),
                action="Check Markets context before opening a new position.",
            )
        )

    if preferences.strategy:
        signal_label = _display_signal(primary_signal)
        severity = "high" if confidence >= 0.8 else "medium" if confidence >= 0.55 else "low"
        if primary_signal == "dont_buy":
            title = f"{symbol}: Avoid entry signal"
            action = "Avoid opening a new long position until conditions improve."
        elif primary_signal == "buy":
            title = f"{symbol}: Buy signal"
            action = "Review strategy page before entering; confirm risk settings."
        else:
            title = f"{symbol}: Hold signal"
            action = "Monitor the asset and wait for clearer confirmation."

        forecast_bit = f" Forecast {predicted_return * 100:+.2f}%"
        if predicted_price:
            forecast_bit += f" (target {predicted_price:,.2f})"
        alerts.append(
            _make_alert(
                alert_id=f"{symbol}-strategy-signal",
                symbol=symbol,
                category="strategy",
                severity=severity,
                title=title,
                message=(
                    f"Price {latest_close:,.2f} ({return_24h:+.2f}% 24h). "
                    f"Strategy: {strategy_display_name} on {strategy_timeframe}. "
                    f"Confidence {confidence * 100:.1f}%.{forecast_bit}. "
                    f"Indicator: {_display_signal(indicator_signal)}."
                ),
                action=action,
            )
        )

    if preferences.sentiment:
        if sentiment_source == "unavailable":
            alerts.append(
                _make_alert(
                    alert_id=f"{symbol}-sentiment-unavailable",
                    symbol=symbol,
                    category="sentiment",
                    severity="low",
                    title=f"{symbol} sentiment unavailable",
                    message="No usable sentiment layer is currently available for this asset.",
                    action="Use market and futures context only for this asset.",
                )
            )
        elif sentiment_source == "gdelt_asset_news":
            severity = "medium" if gdelt_label in {"supportive", "risk_off"} else "low"
            if gdelt_risk_theme in {"flows", "security", "regulation", "macro_stress"}:
                severity = "high" if gdelt_label == "risk_off" else severity
            alerts.append(
                _make_alert(
                    alert_id=f"{symbol}-gdelt-sentiment",
                    symbol=symbol,
                    category="sentiment",
                    severity=severity,
                    title=f"{symbol} news sentiment is {_display_signal(gdelt_label)}",
                    message=(
                        f"Asset-specific GDELT news sentiment is {gdelt_label}. "
                        f"The effective sentiment layer is {sentiment_label}. "
                        f"Dominant theme is {_display_label(gdelt_theme)}, risk theme is {_display_label(gdelt_risk_theme)}, "
                        f"and supportive theme is {_display_label(gdelt_supportive_theme)}."
                    ),
                    action=(
                        "Use as a confirmation layer, not as a standalone signal."
                        if gdelt_risk_theme == "none"
                        else f"Review the active {_display_label(gdelt_risk_theme)} news theme before acting."
                    ),
                )
            )
        else:
            alerts.append(
                _make_alert(
                    alert_id=f"{symbol}-sentiment-fallback",
                    symbol=symbol,
                    category="sentiment",
                    severity="low",
                    title=f"{symbol} uses broad mood fallback",
                    message=(
                        "Asset-specific sentiment is not active, so LiveStrat is using "
                        f"{sentiment_source.replace('_', ' ')}."
                    ),
                    action="Treat sentiment as broad market context only.",
                )
            )

    if preferences.onchain:
        if onchain_label != "unavailable":
            severity = "medium" if onchain_label in {"supportive", "weakening"} else "low"
            if onchain_risk_driver in {"distribution_risk", "trend_divergence", "valuation_stretch"}:
                severity = "high" if onchain_label != "supportive" else "medium"
            alerts.append(
                _make_alert(
                    alert_id=f"{symbol}-onchain",
                    symbol=symbol,
                    category="onchain",
                    severity=severity,
                    title=f"{symbol} on-chain context is {_display_signal(onchain_label)}",
                    message=(
                        f"The latest on-chain regime is {onchain_label}. "
                        f"Primary support driver is {_display_label(onchain_support_driver)} and "
                        f"primary risk driver is {_display_label(onchain_risk_driver)}. "
                        f"Participation breadth is {onchain_participation_breadth:.2f} and structural fragility is "
                        f"{onchain_structural_fragility:.2f}. "
                        "This is slower structural context rather than intraday timing."
                    ),
                    action="Use on-chain context as confirmation for strategy selection.",
                )
            )
        elif onchain_status == "stale":
            alerts.append(
                _make_alert(
                    alert_id=f"{symbol}-onchain-stale",
                    symbol=symbol,
                    category="onchain",
                    severity="low",
                    title=f"{symbol} on-chain context is stale",
                    message="The latest available on-chain snapshot is stale, so it should not drive current decisions.",
                    action="Prioritise market and futures context until on-chain data refreshes.",
                )
            )

    # context mix alert omitted on purpose. It is housekeeping, not a trade signal.
    # The Analytics page already shows the active context family.

    return alerts


def build_alert_response(
    symbol: str,
    summary: dict[str, Any],
    raw_preferences: dict[str, Any] | None = None,
) -> dict[str, Any]:
    preferences = parse_alert_preferences(raw_preferences)
    alerts = build_alerts_for_market_summary(symbol, summary, preferences)
    primary_alert = alerts[0] if alerts else None
    return {
        "symbol": symbol,
        "alert_count": len(alerts),
        "primary_alert": primary_alert,
        "alerts": alerts,
    }
