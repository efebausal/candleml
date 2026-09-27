import io
import json
import sqlite3
import ssl
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError
from urllib.request import Request

import jev_forecast as engine


UTC = timezone.utc
EASTERN = engine.EASTERN


def make_candles(
    count: int,
    *,
    start: datetime | None = None,
    base_price: float = 500.0,
    step: float = 0.05,
) -> list[engine.Candle]:
    start = start or datetime(2025, 3, 10, 9, 30, tzinfo=EASTERN)
    candles = []
    for index in range(count):
        open_price = base_price + index * step
        close = open_price + step * 0.7
        candles.append(
            engine.Candle(
                timestamp=(start + timedelta(minutes=15 * index)).astimezone(UTC),
                open=open_price,
                high=close + 0.08,
                low=open_price - 0.08,
                close=close,
                volume=1_000_000 + index * 1_000,
            )
        )
    return candles


def atr_choice_answers(
    take_profit: str = "2.5 ATR",
    stop_loss: str = "0.75 ATR",
) -> dict[str, dict[str, str]]:
    return {
        "take_profit_atr": {"type": "choice", "choice": take_profit},
        "stop_loss_atr": {"type": "choice", "choice": stop_loss},
    }


class TwelveDataParsingTests(unittest.TestCase):
    def test_ssl_context_requires_certificate_verification(self) -> None:
        context = engine._ssl_context()
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)
        self.assertTrue(context.get_ca_certs())

    def test_parses_eastern_time_and_drops_forming_bar(self) -> None:
        payload = {
            "status": "ok",
            "values": [
                {"datetime": "2025-03-10 09:30:00", "open": "500", "high": "501", "low": "499", "close": "500.5", "volume": "10"},
                {"datetime": "2025-03-10 09:45:00", "open": "500.5", "high": "502", "low": "500", "close": "501", "volume": "20"},
                {"datetime": "2025-03-10 10:00:00", "open": "501", "high": "503", "low": "500.5", "close": "502", "volume": "30"},
            ],
        }
        now = datetime(2025, 3, 10, 10, 2, tzinfo=EASTERN)

        candles = engine.parse_twelvedata_payload(payload, now=now)

        self.assertEqual(len(candles), 2)
        self.assertEqual(candles[0].timestamp, datetime(2025, 3, 10, 13, 30, tzinfo=UTC))
        self.assertEqual(candles[-1].close, 501.0)

    def test_parses_crypto_bars_in_utc_and_keeps_exchange_metadata(self) -> None:
        payload = {
            "meta": {"symbol": "BTC/USD", "exchange": "Binance"},
            "values": [
                {"datetime": "2025-03-08 10:00:00", "open": 90000, "high": 90100, "low": 89900, "close": 90050, "volume": 12},
                {"datetime": "2025-03-08 10:15:00", "open": 90050, "high": 90200, "low": 90000, "close": 90150, "volume": 8},
            ],
        }

        candles = engine.parse_twelvedata_payload(
            payload,
            now=datetime(2025, 3, 8, 10, 17, tzinfo=UTC),
            timezone_name="UTC",
            symbol="BTC/USD",
        )

        self.assertEqual(len(candles), 1)
        self.assertEqual(candles[0].timestamp, datetime(2025, 3, 8, 10, 0, tzinfo=UTC))
        self.assertEqual(candles[0].exchange, "Binance")

    def test_fetch_crypto_bars_requests_utc_and_optional_exchange(self) -> None:
        payload = {
            "meta": {"exchange": "Binance"},
            "values": [
                {"datetime": "2025-03-08 10:00:00", "open": 90000, "high": 90100, "low": 89900, "close": 90050, "volume": 12}
            ],
        }
        with patch.object(engine, "_http_json", return_value=payload) as mocked:
            candles = engine.fetch_twelvedata_bars(
                "unit-test-twelve-data-key",
                symbol="BTC/USD",
                timezone_name="UTC",
                exchange="Binance",
                now=datetime(2025, 3, 8, 10, 16, tzinfo=UTC),
            )

        request = mocked.call_args.args[0]
        query = parse_qs(urlparse(request.full_url).query)
        self.assertEqual(query["symbol"], ["BTC/USD"])
        self.assertEqual(query["timezone"], ["UTC"])
        self.assertEqual(query["exchange"], ["Binance"])
        self.assertEqual(candles[0].exchange, "Binance")

    def test_fetch_spy_keeps_eastern_timezone_default(self) -> None:
        payload = {
            "values": [
                {"datetime": "2025-03-10 09:30:00", "open": 500, "high": 501, "low": 499, "close": 500.5, "volume": 10}
            ],
        }
        with patch.object(engine, "_http_json", return_value=payload) as mocked:
            engine.fetch_twelvedata_bars(
                "unit-test-twelve-data-key",
                now=datetime(2025, 3, 10, 10, 0, tzinfo=EASTERN),
            )

        request = mocked.call_args.args[0]
        query = parse_qs(urlparse(request.full_url).query)
        self.assertEqual(query["symbol"], ["SPY"])
        self.assertEqual(query["timezone"], ["America/New_York"])
        self.assertNotIn("exchange", query)

    def test_rejects_provider_error_payload(self) -> None:
        with self.assertRaisesRegex(engine.ForecastError, "API key limit"):
            engine.parse_twelvedata_payload({"status": "error", "message": "API key limit reached"})

    def test_http_error_includes_safe_detail_without_echoing_request_url(self) -> None:
        secret = "not-a-real-secret"
        error = HTTPError(
            url=f"https://api.example.test/endpoint?apikey={secret}",
            code=401,
            msg="Unauthorized",
            hdrs=None,
            fp=io.BytesIO(json.dumps({"detail": "authentication_error: invalid API key"}).encode()),
        )
        request = Request(f"https://api.example.test/endpoint?apikey={secret}")
        with patch.object(engine, "urlopen", side_effect=error):
            with self.assertRaises(engine.ForecastError) as context:
                engine._http_json(request, service="OpenRouter Jev", timeout=1)

        self.assertIn("HTTP 401", str(context.exception))
        self.assertIn("invalid API key", str(context.exception))
        self.assertNotIn(secret, str(context.exception))

    def test_rejects_inconsistent_ohlc(self) -> None:
        payload = {
            "values": [
                {"datetime": "2025-03-10 09:30:00", "open": 500, "high": 499, "low": 498, "close": 500, "volume": 10}
            ]
        }
        with self.assertRaisesRegex(engine.ForecastError, "inconsistent OHLC"):
            engine.parse_twelvedata_payload(
                payload,
                now=datetime(2025, 3, 10, 10, 0, tzinfo=EASTERN),
            )


class ForecastLogicTests(unittest.TestCase):
    def test_forecast_uses_and_persists_jev_selected_atr_levels(self) -> None:
        current = datetime.now(UTC)
        candles = make_candles(
            64,
            start=current - timedelta(minutes=15 * 64 + 10),
            base_price=500.0,
            step=0.05,
        )
        args = engine.build_parser().parse_args(["forecast", "--market", "crypto"])
        store_mock = MagicMock()
        store_mock.resolve_pending.return_value = 0
        store_mock.find_cached.return_value = None
        store_mock.save_forecast.return_value = 12
        answer = engine.JevAnswer(
            signal="LONG",
            probabilities={"LONG": 0.7, "FLAT": 0.2, "SHORT": 0.1},
            confidence=0.7,
            model="jev-test",
            input_tokens=100,
            output_tokens=20,
            take_profit_atr=2.5,
            stop_loss_atr=0.75,
        )
        output = io.StringIO()

        with (
            patch.object(engine, "fetch_twelvedata_bars", return_value=candles),
            patch.object(engine, "ForecastStore", return_value=store_mock),
            patch.object(engine, "openrouter_configuration", return_value=("test-key", "jev-test")),
            patch.object(engine, "call_jev", return_value=answer),
            redirect_stdout(output),
        ):
            result = engine._forecast_command(args)

        self.assertEqual(result, 0)
        saved = store_mock.save_forecast.call_args.kwargs
        self.assertEqual(saved["take_profit_atr"], 2.5)
        self.assertEqual(saved["stop_loss_atr"], 0.75)
        self.assertEqual(saved["atr_choice_config_json"], engine.ATR_CHOICE_CONFIG_JSON)
        self.assertGreater(saved["take_profit_price"], candles[-1].close)
        self.assertLess(saved["stop_loss_price"], candles[-1].close)
        self.assertIn("2.5 x ATR", output.getvalue())
        self.assertIn("0.75 x ATR", output.getvalue())

    def test_crypto_cli_dry_run_uses_btc_defaults_and_separate_store(self) -> None:
        current = datetime.now(UTC)
        candles = make_candles(
            64,
            start=current - timedelta(minutes=15 * 64 + 10),
            base_price=90000,
            step=10,
        )
        args = engine.build_parser().parse_args(["forecast", "--market", "crypto", "--dry-run"])
        output = io.StringIO()
        store_mock = MagicMock()
        store_mock.resolve_pending.return_value = 0

        with (
            patch.object(engine, "fetch_twelvedata_bars", return_value=candles) as fetch,
            patch.object(engine, "ForecastStore", return_value=store_mock) as store_type,
            redirect_stdout(output),
        ):
            result = engine._forecast_command(args)

        self.assertEqual(result, 0)
        self.assertEqual(fetch.call_args.kwargs["symbol"], "BTC/USD")
        self.assertEqual(fetch.call_args.kwargs["timezone_name"], "UTC")
        self.assertEqual(store_type.call_args.args[0].name, "btc_forecasts.sqlite3")
        state_json = output.getvalue().split("\nDry run:", 1)[0]
        state = json.loads(state_json)
        self.assertEqual(state["instrument"]["market"], "crypto")
        self.assertEqual(state["instrument"]["symbol"], "BTC/USD")

    def test_ledger_symbol_separates_exchanges(self) -> None:
        candle = engine.Candle(
            datetime(2025, 3, 8, 10, 0, tzinfo=UTC),
            90000,
            90100,
            89900,
            90050,
            exchange="Binance",
        )
        coinbase_candle = engine.Candle(
            candle.timestamp,
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            exchange="Coinbase",
        )

        self.assertEqual(engine._ledger_symbol("BTC/USD", None, [candle]), "BTC/USD@Binance")
        self.assertEqual(engine._ledger_symbol("BTC/USD", "Coinbase", [candle]), "BTC/USD@Binance")
        self.assertEqual(engine._ledger_symbol("BTC/USD", None, [coinbase_candle]), "BTC/USD@Coinbase")

    def test_features_and_state_use_completed_candles(self) -> None:
        candles = make_candles(40)
        state = engine.build_market_state(candles, state_bars=32)

        self.assertEqual(len(state["recent_completed_bars"]), 32)
        self.assertEqual(state["last_completed_bar"]["closed_at_utc"], engine._iso_utc(candles[-1].close_time))
        self.assertAlmostEqual(state["features"]["return_15m_bps"], engine._return_bps(candles[-2].close, candles[-1].close), places=4)
        self.assertIsNotNone(state["features"]["volume_ratio_4_to_20"])

    def test_crypto_market_state_identifies_utc_provider_and_cfd_basis(self) -> None:
        candles = [
            engine.Candle(
                timestamp=datetime(2025, 3, 8, 10, 0, tzinfo=UTC) + timedelta(minutes=15 * index),
                open=90000 + index,
                high=90002 + index,
                low=89999 + index,
                close=90001 + index,
                volume=10 + index,
                exchange="Binance",
            )
            for index in range(40)
        ]

        state = engine.build_market_state(candles, symbol="BTC/USD", market="crypto")

        self.assertEqual(state["instrument"]["bar_timezone"], "UTC")
        self.assertEqual(state["instrument"]["market"], "crypto")
        self.assertEqual(state["instrument"]["data_provider"], "Twelve Data")
        self.assertEqual(state["instrument"]["exchange"], "Binance")
        self.assertIn("may differ from the FTMO BTCUSD CFD quote", state["instrument"]["description"])

    def test_jev_request_defines_all_three_outcomes(self) -> None:
        state = engine.build_market_state(make_candles(32), symbol="SPY", horizon_bars=4, neutral_band_bps=10)
        request = engine.build_jev_request(state)
        criteria = request["questions"]["direction"]["criteria"]

        self.assertEqual(set(criteria), {"LONG", "FLAT", "SHORT"})
        self.assertIn("10 basis points", criteria["FLAT"])
        self.assertEqual(request["model"], "~typesafe/jev-latest")
        self.assertEqual(
            set(request["questions"]["take_profit_atr"]["criteria"]),
            {f"{multiple:g} ATR" for multiple in engine.TAKE_PROFIT_ATR_CHOICES},
        )
        self.assertEqual(
            set(request["questions"]["stop_loss_atr"]["criteria"]),
            {f"{multiple:g} ATR" for multiple in engine.STOP_LOSS_ATR_CHOICES},
        )

    def test_openrouter_configuration_reads_key_and_model(self) -> None:
        api_key, model = engine.openrouter_configuration(
            {"OPENROUTER_API_KEY": "local-key", "OPENROUTER_MODEL": "typesafe/jev-1.13"}
        )
        self.assertEqual(api_key, "local-key")
        self.assertEqual(model, "typesafe/jev-1.13")

    def test_call_jev_uses_openrouter_decisions_endpoint(self) -> None:
        payload = {
            "model": "typesafe/jev-1.13-20260917",
            "answers": {
                "direction": {
                    "type": "choice",
                    "choice": "LONG",
                    "probabilities": {"LONG": 0.7, "FLAT": 0.2, "SHORT": 0.1},
                    "confidence": 0.6,
                },
                **atr_choice_answers(),
            },
            "usage": {"input_tokens": 300, "output_tokens": 40, "cost": 0.0000126},
        }
        state = engine.build_market_state(make_candles(32))
        with patch.object(engine, "urlopen", return_value=io.BytesIO(json.dumps(payload).encode())) as mocked:
            answer = engine.call_jev("unit-test-openrouter-key", state)

        request = mocked.call_args.args[0]
        self.assertEqual(request.full_url, "https://openrouter.ai/api/alpha/decisions")
        self.assertEqual(request.get_header("Authorization"), "Bearer unit-test-openrouter-key")
        request_body = json.loads(request.data)
        self.assertEqual(request_body["model"], "~typesafe/jev-latest")
        self.assertEqual(answer.signal, "LONG")
        self.assertEqual(answer.take_profit_atr, 2.5)
        self.assertEqual(answer.stop_loss_atr, 0.75)
        self.assertAlmostEqual(answer.cost_usd or 0.0, 0.0000126)
        self.assertIn("take_profit_atr", request_body["questions"])
        self.assertIn("stop_loss_atr", request_body["questions"])

    def test_validates_jev_choice_probabilities(self) -> None:
        answer = engine.parse_jev_response(
            {
                "result": {
                    "model": "jev-1.13.0",
                    "answers": {
                        "direction": {
                            "type": "choice",
                            "choice": "LONG",
                            "probabilities": {"LONG": 0.7, "FLAT": 0.2, "SHORT": 0.1},
                            "confidence": 0.6,
                        },
                        **atr_choice_answers(),
                    },
                    "usage": {"input_tokens": 300, "output_tokens": 40},
                },
                "elapsedMs": 99,
            }
        )

        self.assertEqual(answer.signal, "LONG")
        self.assertEqual(answer.model, "jev-1.13.0")
        self.assertEqual(answer.input_tokens, 300)
        self.assertAlmostEqual(sum(answer.probabilities.values()), 1.0)

    def test_supports_unwrapped_openrouter_response(self) -> None:
        answer = engine.parse_jev_response(
            {
                "model": "jev-1.13.0",
                "answers": {
                    "direction": {
                        "type": "choice",
                        "choice": "FLAT",
                        "probabilities": {"LONG": 0.2, "FLAT": 0.6, "SHORT": 0.2},
                        "confidence": 0.4,
                    },
                    **atr_choice_answers("1 ATR", "0.5 ATR"),
                },
                "usage": {"input_tokens": 100, "output_tokens": 12},
            }
        )
        self.assertEqual(answer.signal, "FLAT")

    def test_supports_deeply_wrapped_jev_response(self) -> None:
        answer = engine.parse_jev_response(
            {
                "result": {
                    "data": {
                        "model": "jev-1.13.0",
                        "answers": {
                            "direction": {
                                "type": "choice",
                                "choice": "SHORT",
                                "probabilities": {"LONG": 0.1, "FLAT": 0.2, "SHORT": 0.7},
                                "confidence": 0.6,
                            },
                            **atr_choice_answers("1.5 ATR", "1 ATR"),
                        },
                    },
                    "usage": {"input_tokens": 100, "output_tokens": 12},
                }
            }
        )
        self.assertEqual(answer.signal, "SHORT")
        self.assertEqual(answer.input_tokens, 100)

    def test_unrecognized_jev_shape_reports_only_keys(self) -> None:
        with self.assertRaisesRegex(engine.ForecastError, r"level0=\[elapsedMs,result\].*level1=\[code,error\]") as context:
            engine.parse_jev_response({"elapsedMs": 10, "result": {"code": "bad", "error": "private value"}})
        self.assertNotIn("private value", str(context.exception))

    def test_rejects_inconsistent_jev_choice(self) -> None:
        payload = {
            "model": "jev-latest",
            "answers": {
                "direction": {
                    "type": "choice",
                    "choice": "SHORT",
                    "probabilities": {"LONG": 0.7, "FLAT": 0.2, "SHORT": 0.1},
                    "confidence": 0.6,
                },
                **atr_choice_answers(),
            },
        }
        with self.assertRaisesRegex(engine.ForecastError, "does not match"):
            engine.parse_jev_response(payload)

    def test_rejects_atr_choice_outside_configured_ladder(self) -> None:
        payload = {
            "model": "jev-latest",
            "answers": {
                "direction": {
                    "type": "choice",
                    "choice": "LONG",
                    "probabilities": {"LONG": 0.7, "FLAT": 0.2, "SHORT": 0.1},
                },
                **atr_choice_answers("2 ATR", "2 ATR"),
            },
        }
        with self.assertRaisesRegex(engine.ForecastError, "unsupported stop_loss_atr choice"):
            engine.parse_jev_response(payload)

    def test_classification_deadband(self) -> None:
        self.assertEqual(engine.classify_return(0.11, 10), "LONG")
        self.assertEqual(engine.classify_return(0.10, 10), "FLAT")
        self.assertEqual(engine.classify_return(-0.10, 10), "FLAT")
        self.assertEqual(engine.classify_return(-0.11, 10), "SHORT")

    def test_rejects_horizon_that_crosses_regular_close(self) -> None:
        candles = make_candles(23, start=datetime(2025, 3, 10, 9, 30, tzinfo=EASTERN))
        # The last bar closes at 15:15 ET; a one-hour horizon would end after 16:00.
        now = candles[-1].close_time + timedelta(minutes=1)
        with self.assertRaisesRegex(engine.ForecastError, "Not enough regular-session time"):
            engine.validate_forecast_window(candles, horizon_bars=4, now=now)

    def test_accepts_one_hour_window_during_regular_session(self) -> None:
        candles = make_candles(11, start=datetime(2025, 3, 10, 9, 30, tzinfo=EASTERN))
        now = candles[-1].close_time + timedelta(minutes=1)
        decision = engine.validate_forecast_window(candles, horizon_bars=4, now=now)
        self.assertEqual(decision, candles[-1].close_time)

    def test_crypto_forecast_window_accepts_weekend_bars(self) -> None:
        candles = make_candles(
            11,
            start=datetime(2025, 3, 8, 10, 0, tzinfo=UTC),
            base_price=90000,
            step=10,
        )
        now = candles[-1].close_time + timedelta(minutes=1)

        decision = engine.validate_forecast_window(
            candles,
            horizon_bars=4,
            now=now,
            market="crypto",
            symbol="BTC/USD",
        )

        self.assertEqual(decision, candles[-1].close_time)

    def test_rejects_stale_crypto_bars_with_symbol_in_error(self) -> None:
        candles = make_candles(11, start=datetime(2025, 3, 8, 10, 0, tzinfo=UTC))
        now = candles[-1].close_time + timedelta(minutes=31)

        with self.assertRaisesRegex(engine.ForecastError, r"BTC/USD candle is 31 minutes old"):
            engine.validate_forecast_window(
                candles,
                horizon_bars=4,
                now=now,
                market="crypto",
                symbol="BTC/USD",
            )

    def test_rejects_stale_market_data_by_default(self) -> None:
        candles = make_candles(11, start=datetime(2025, 3, 10, 9, 30, tzinfo=EASTERN))
        now = candles[-1].close_time + timedelta(minutes=31)
        with self.assertRaisesRegex(engine.ForecastError, "minutes old"):
            engine.validate_forecast_window(candles, horizon_bars=4, now=now)

    def test_explains_stale_data_when_market_is_closed(self) -> None:
        candles = make_candles(1, start=datetime(2025, 3, 10, 14, 45, tzinfo=EASTERN))
        now = datetime(2025, 3, 10, 21, 39, tzinfo=EASTERN)
        with self.assertRaisesRegex(engine.ForecastError, "session is closed.*pre/post-market"):
            engine.validate_forecast_window(candles, horizon_bars=4, now=now)

    def test_local_env_file_loads_values_without_overriding_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            env_path = Path(temporary_directory) / ".env"
            env_path.write_text(
                '# local keys\nTWELVEDATA_API_KEY="from-file"\nexport OTHER_TEST_KEY=from-file-too\n',
                encoding="utf-8",
            )
            environ = {"TWELVEDATA_API_KEY": "already-set"}

            engine.load_env_file(env_path, environ)

            self.assertEqual(environ["TWELVEDATA_API_KEY"], "already-set")
            self.assertEqual(environ["OTHER_TEST_KEY"], "from-file-too")


class ForecastStoreTests(unittest.TestCase):
    def test_crypto_default_database_is_separate_from_spy(self) -> None:
        self.assertEqual(engine._market_db_path("crypto").name, "btc_forecasts.sqlite3")
        self.assertEqual(engine._market_db_path("equity").name, "jev_forecasts.sqlite3")

    def test_cache_and_resolve_forward_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = engine.ForecastStore(Path(temporary_directory) / "forecasts.sqlite3")
            first = make_candles(1)[0]
            decision_time = first.close_time
            answer = engine.JevAnswer(
                signal="LONG",
                probabilities={"LONG": 0.7, "FLAT": 0.2, "SHORT": 0.1},
                confidence=0.6,
                model="jev-1.13.0",
                input_tokens=300,
                output_tokens=40,
            )
            state = {"instrument": {"symbol": "SPY"}}
            store.save_forecast(
                symbol="SPY",
                interval="15min",
                decision_time=decision_time,
                decision_close=first.close,
                horizon_bars=4,
                neutral_band_bps=10,
                lookback_bars=64,
                requested_model="jev-latest",
                answer=answer,
                state=state,
                requested_at=decision_time,
            )
            cached = store.find_cached(
                symbol="SPY",
                interval="15min",
                decision_time=decision_time,
                horizon_bars=4,
                neutral_band_bps=10,
                lookback_bars=64,
                requested_model="jev-latest",
            )
            self.assertIsNotNone(cached)

            target_start = decision_time + timedelta(minutes=45)
            target = engine.Candle(
                timestamp=target_start,
                open=first.close,
                high=first.close + 1.2,
                low=first.close,
                close=first.close + 1.0,
                volume=2_000_000,
            )
            resolved = store.resolve_pending([target], symbol="SPY", interval="15min", now=target.close_time)
            self.assertEqual(resolved, 1)
            rows = store.list_forecasts()
            self.assertEqual(rows[0]["realized_class"], "LONG")
            self.assertAlmostEqual(rows[0]["realized_return_pct"], (target.close / first.close - 1) * 100)

    def test_resolution_does_not_cross_symbols(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = engine.ForecastStore(Path(temporary_directory) / "forecasts.sqlite3")
            first = make_candles(1)[0]
            answer = engine.JevAnswer("LONG", {"LONG": 0.7, "FLAT": 0.2, "SHORT": 0.1}, 0.6, "jev-1.13.0", 300, 40)
            store.save_forecast(
                symbol="QQQ",
                interval="15min",
                decision_time=first.close_time,
                decision_close=first.close,
                horizon_bars=4,
                neutral_band_bps=10,
                lookback_bars=64,
                requested_model="jev-latest",
                answer=answer,
                state={"instrument": {"symbol": "QQQ"}},
            )
            target = engine.Candle(first.close_time + timedelta(minutes=45), 500, 502, 500, 501, 10)
            self.assertEqual(store.resolve_pending([target], symbol="SPY", interval="15min"), 0)
            self.assertIsNone(store.list_forecasts()[0]["realized_class"])

    def _save_trade_forecast(
        self,
        store: engine.ForecastStore,
        first: engine.Candle,
        *,
        signal: str = "LONG",
        horizon_bars: int = 3,
        take_profit_price: float | None = None,
        stop_loss_price: float | None = None,
    ) -> None:
        entry = first.close
        if take_profit_price is None and signal == "LONG":
            take_profit_price = entry + 1.0
        if stop_loss_price is None and signal == "LONG":
            stop_loss_price = entry - 0.5
        if take_profit_price is None and signal == "SHORT":
            take_profit_price = entry - 1.0
        if stop_loss_price is None and signal == "SHORT":
            stop_loss_price = entry + 0.5
        probabilities = {
            "LONG": 0.8 if signal == "LONG" else 0.1,
            "FLAT": 0.6 if signal == "FLAT" else 0.1,
            "SHORT": 0.8 if signal == "SHORT" else 0.1,
        }
        answer = engine.JevAnswer(signal, probabilities, 0.8, "jev-test", 100, 20)
        store.save_forecast(
            symbol="SPY",
            interval="15min",
            decision_time=first.close_time,
            decision_close=entry,
            horizon_bars=horizon_bars,
            neutral_band_bps=10,
            lookback_bars=64,
            requested_model="jev-test",
            answer=answer,
            state={"features": {"atr_14_bps": 10}},
            atr_price=0.5,
            take_profit_atr=2.0,
            stop_loss_atr=1.0,
            atr_choice_config_json=engine.ATR_CHOICE_CONFIG_JSON,
            take_profit_price=take_profit_price,
            stop_loss_price=stop_loss_price,
        )

    @staticmethod
    def _future_bar(
        timestamp: datetime,
        *,
        entry: float,
        high: float,
        low: float,
        close: float,
    ) -> engine.Candle:
        return engine.Candle(timestamp, entry, high, low, close, 100)

    def test_tracks_early_take_profit_and_keeps_horizon_classification_pending(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = engine.ForecastStore(Path(temporary_directory) / "forecasts.sqlite3")
            first = make_candles(1)[0]
            entry = first.close
            decision_time = first.close_time
            target = entry + 1.0
            self._save_trade_forecast(store, first, take_profit_price=target)
            touch_bar = self._future_bar(
                decision_time,
                entry=entry,
                high=target + 0.1,
                low=entry - 0.1,
                close=entry + 0.2,
            )

            self.assertEqual(store.resolve_pending([touch_bar], symbol="SPY", interval="15min"), 0)
            row = store.list_forecasts()[0]
            self.assertEqual(row["trade_outcome"], "TAKE_PROFIT")
            self.assertAlmostEqual(row["trade_exit_price"], target)
            self.assertIsNone(row["realized_class"])

            middle_bar = self._future_bar(
                decision_time + timedelta(minutes=15),
                entry=entry,
                high=entry + 0.2,
                low=entry - 0.2,
                close=entry + 0.1,
            )
            horizon_bar = self._future_bar(
                decision_time + timedelta(minutes=30),
                entry=entry,
                high=entry + 0.3,
                low=entry - 0.2,
                close=entry + 0.2,
            )
            self.assertEqual(
                store.resolve_pending(
                    [touch_bar, middle_bar, horizon_bar], symbol="SPY", interval="15min"
                ),
                1,
            )
            row = store.list_forecasts()[0]
            self.assertEqual(row["trade_outcome"], "TAKE_PROFIT")
            self.assertIsNotNone(row["realized_class"])

    def test_same_bar_target_and_stop_uses_stop_first(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = engine.ForecastStore(Path(temporary_directory) / "forecasts.sqlite3")
            first = make_candles(1)[0]
            entry = first.close
            decision_time = first.close_time
            target = entry + 1.0
            stop = entry - 0.5
            self._save_trade_forecast(
                store,
                first,
                horizon_bars=1,
                take_profit_price=target,
                stop_loss_price=stop,
            )
            bar = self._future_bar(
                decision_time,
                entry=entry,
                high=target + 0.1,
                low=stop - 0.1,
                close=entry,
            )

            self.assertEqual(store.resolve_pending([bar], symbol="SPY", interval="15min"), 1)
            row = store.list_forecasts()[0]
            self.assertEqual(row["trade_outcome"], "STOP_LOSS")
            self.assertLess(row["trade_return_pct"], 0)

    def test_short_take_profit_and_horizon_exit_are_direction_adjusted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = engine.ForecastStore(Path(temporary_directory) / "forecasts.sqlite3")
            first = make_candles(1)[0]
            entry = first.close
            decision_time = first.close_time
            target = entry - 1.0
            self._save_trade_forecast(
                store,
                first,
                signal="SHORT",
                horizon_bars=1,
                take_profit_price=target,
            )
            bar = self._future_bar(
                decision_time,
                entry=entry,
                high=entry + 0.1,
                low=target - 0.1,
                close=entry - 0.2,
            )

            store.resolve_pending([bar], symbol="SPY", interval="15min")
            row = store.list_forecasts()[0]
            self.assertEqual(row["trade_outcome"], "TAKE_PROFIT")
            self.assertGreater(row["trade_return_pct"], 0)

    def test_no_touch_uses_horizon_close_and_flat_signal_is_not_traded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = engine.ForecastStore(Path(temporary_directory) / "forecasts.sqlite3")
            first = make_candles(1)[0]
            entry = first.close
            decision_time = first.close_time
            self._save_trade_forecast(store, first, horizon_bars=2)
            bars = [
                self._future_bar(
                    decision_time + timedelta(minutes=15 * index),
                    entry=entry,
                    high=entry + 0.2,
                    low=entry - 0.2,
                    close=entry + 0.1,
                )
                for index in range(2)
            ]
            store.resolve_pending(bars, symbol="SPY", interval="15min")
            row = store.list_forecasts()[0]
            self.assertEqual(row["trade_outcome"], "HORIZON")
            self.assertAlmostEqual(row["trade_exit_price"], bars[-1].close)

            flat_store = engine.ForecastStore(Path(temporary_directory) / "flat.sqlite3")
            self._save_trade_forecast(flat_store, first, signal="FLAT", horizon_bars=1)
            flat_bar = self._future_bar(
                decision_time,
                entry=entry,
                high=entry + 0.2,
                low=entry - 0.2,
                close=entry,
            )
            flat_store.resolve_pending([flat_bar], symbol="SPY", interval="15min")
            self.assertEqual(flat_store.list_forecasts()[0]["trade_outcome"], "NO_TRADE")

    def test_cache_identity_includes_atr_choice_ladder(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            store = engine.ForecastStore(Path(temporary_directory) / "forecasts.sqlite3")
            first = make_candles(1)[0]
            self._save_trade_forecast(store, first)
            query = {
                "symbol": "SPY",
                "interval": "15min",
                "decision_time": first.close_time,
                "horizon_bars": 3,
                "neutral_band_bps": 10,
                "lookback_bars": 64,
                "requested_model": "jev-test",
                "atr_choice_config_json": engine.ATR_CHOICE_CONFIG_JSON,
            }
            self.assertIsNotNone(store.find_cached(**query))
            self.assertIsNone(
                store.find_cached(
                    **{**query, "atr_choice_config_json": '{"take_profit_atr":[1.0]}'}
                )
            )

    def test_migrates_existing_forecast_database(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "legacy.sqlite3"
            connection = sqlite3.connect(path)
            try:
                connection.execute(
                    """
                    CREATE TABLE forecasts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        decision_time_utc TEXT NOT NULL,
                        decision_close REAL NOT NULL,
                        horizon_bars INTEGER NOT NULL,
                        neutral_band_bps REAL NOT NULL,
                        lookback_bars INTEGER NOT NULL,
                        requested_model TEXT NOT NULL,
                        signal TEXT NOT NULL,
                        probabilities_json TEXT NOT NULL,
                        confidence REAL,
                        resolved_model TEXT NOT NULL,
                        input_tokens INTEGER,
                        output_tokens INTEGER,
                        input_state_json TEXT NOT NULL,
                        requested_at_utc TEXT NOT NULL,
                        realized_return_pct REAL,
                        realized_class TEXT,
                        resolved_at_utc TEXT
                    )
                    """
                )
            finally:
                connection.close()

            engine.ForecastStore(path)
            connection = sqlite3.connect(path)
            try:
                columns = {row[1] for row in connection.execute("PRAGMA table_info(forecasts)")}
            finally:
                connection.close()
            self.assertTrue(
                {
                    "cost_usd",
                    "take_profit_atr",
                    "stop_loss_atr",
                    "atr_choice_config_json",
                    "take_profit_price",
                    "stop_loss_price",
                    "trade_outcome",
                    "trade_exit_time_utc",
                }.issubset(columns)
            )


class TradeLevelTests(unittest.TestCase):
    def test_levels_are_oriented_by_signal_and_flat_has_none(self) -> None:
        long_levels = engine.calculate_trade_levels(
            entry_price=100.0,
            signal="LONG",
            atr_14_bps=100.0,
            take_profit_atr=2.0,
            stop_loss_atr=1.0,
        )
        short_levels = engine.calculate_trade_levels(
            entry_price=100.0,
            signal="SHORT",
            atr_14_bps=100.0,
            take_profit_atr=2.0,
            stop_loss_atr=1.0,
        )
        flat_levels = engine.calculate_trade_levels(
            entry_price=100.0,
            signal="FLAT",
            atr_14_bps=100.0,
            take_profit_atr=2.0,
            stop_loss_atr=1.0,
        )

        self.assertEqual(long_levels.atr_price, 1.0)
        self.assertEqual(long_levels.take_profit_price, 102.0)
        self.assertEqual(long_levels.stop_loss_price, 99.0)
        self.assertEqual(short_levels.take_profit_price, 98.0)
        self.assertEqual(short_levels.stop_loss_price, 101.0)
        self.assertIsNone(flat_levels.take_profit_price)
        self.assertIsNone(flat_levels.stop_loss_price)

    def test_jev_selected_multiples_determine_price_levels(self) -> None:
        levels = engine.calculate_trade_levels(
            entry_price=100.0,
            signal="LONG",
            atr_14_bps=100.0,
            take_profit_atr=2.5,
            stop_loss_atr=0.75,
        )

        self.assertEqual(levels.take_profit_price, 102.5)
        self.assertEqual(levels.stop_loss_price, 99.25)


if __name__ == "__main__":
    unittest.main()
