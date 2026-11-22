import json
import os
import pickle
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

IST = ZoneInfo("Asia/Kolkata")
FEATURE_LOG_FILE = "ml_feature_log.json"
MODEL_ARTIFACT_FILE = "ml_context_model.pkl"
MIN_LABELED_SAMPLES = 60

NUMERIC_FEATURES = [
    "side_num",
    "rvol_normalized",
    "vol_accel_5m",
    "vwap_slope_pct",
    "mtf_15m_num",
    "orb_num",
    "clv",
    "rsi",
    "mansfield_rs",
    "sector_chg_pct",
    "sector_alpha",
    "delivery_pct",
    "obi_ratio",
    "bid_ask_spread_pct",
    "vix_ltp",
    "fii_net_cr",
    "brent_chg_pct",
    "us10y_bps_chg",
    "usdinr_chg_pct",
    "nasdaq_chg_pct",
    "is_expiry_num",
]

CATEGORICAL_FEATURES = ["archetype", "entry_session"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


class ContextualMLPredictor:
    """
    Upgrade 3: Logs 20+ market/microstructure/macro features per setup, labels
    outcomes from settled real and shadow trades, trains a walk-forward
    time-series classifier (LightGBM with sklearn fallback), and outputs a
    calibrated win probability + bounded conviction adjustment.
    """

    def __init__(self):
        self.logs = self._load_logs()
        self.artifact = self._load_artifact()

    @staticmethod
    def _now_ist_str() -> str:
        return datetime.now(IST).strftime("%Y-%m-%d %I:%M %p IST")

    def _load_logs(self) -> list:
        if os.path.exists(FEATURE_LOG_FILE):
            try:
                with open(FEATURE_LOG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else []
            except Exception:
                return []
        return []

    def _save_logs(self):
        try:
            with open(FEATURE_LOG_FILE, "w", encoding="utf-8") as f:
                json.dump(self.logs[-1500:], f, indent=2)
        except Exception as e:
            print(f"[ML LOG WARN] Could not save {FEATURE_LOG_FILE}: {e}")

    def _load_artifact(self) -> dict:
        if os.path.exists(MODEL_ARTIFACT_FILE):
            try:
                with open(MODEL_ARTIFACT_FILE, "rb") as f:
                    art = pickle.load(f)
                    if isinstance(art, dict):
                        return art
            except Exception:
                pass
        return {
            "trained": False,
            "samples_count": 0,
            "wf_auc": 0.50,
            "wf_win_rate": 0.0,
            "backend": "NONE",
            "updated_at": "",
            "model": None,
            "cat_maps": {},
        }

    def _save_artifact(self):
        try:
            with open(MODEL_ARTIFACT_FILE, "wb") as f:
                pickle.dump(self.artifact, f)
        except Exception as e:
            print(f"[ML ARTIFACT WARN] Could not save {MODEL_ARTIFACT_FILE}: {e}")

    @staticmethod
    def _encode_features(record: dict) -> dict:
        side_str = str(record.get("action", "BUY")).upper()
        mtf_str = str(record.get("mtf_15m_trend", "NEUTRAL")).upper()
        orb_str = str(record.get("orb_signal", "INSIDE_ORB")).upper()

        side_num = 1.0 if side_str == "BUY" else -1.0
        mtf_num = 1.0 if mtf_str == "BULLISH" else (-1.0 if mtf_str == "BEARISH" else 0.0)
        orb_num = (
            1.0 if (side_str == "BUY" and orb_str == "ORB_BREAKOUT_UP")
            else (-1.0 if (side_str.startswith("SELL") and orb_str == "ORB_BREAKDOWN_DOWN") else 0.0)
        )

        def _f(key, default=0.0):
            try:
                val = record.get(key, default)
                if val is None or val == "":
                    return float(default)
                fval = float(val)
                return float(default) if np.isnan(fval) or np.isinf(fval) else fval
            except Exception:
                return float(default)

        return {
            "side_num": side_num,
            "rvol_normalized": _f("rvol_normalized", 1.0),
            "vol_accel_5m": _f("vol_accel_5m", 1.0),
            "vwap_slope_pct": _f("vwap_slope_pct", 0.0),
            "mtf_15m_num": mtf_num,
            "orb_num": orb_num,
            "clv": _f("clv", 0.0),
            "rsi": _f("rsi", 50.0),
            "mansfield_rs": _f("mansfield_rs", 0.0),
            "sector_chg_pct": _f("sector_chg_pct", 0.0),
            "sector_alpha": _f("sector_alpha", 0.0),
            "delivery_pct": _f("delivery_pct", 0.0),
            "obi_ratio": _f("obi_ratio", 0.0),
            "bid_ask_spread_pct": _f("bid_ask_spread_pct", 0.0),
            "vix_ltp": _f("vix_ltp", 14.0),
            "fii_net_cr": _f("fii_net_cr", 0.0),
            "brent_chg_pct": _f("brent_chg_pct", 0.0),
            "us10y_bps_chg": _f("us10y_bps_chg", 0.0),
            "usdinr_chg_pct": _f("usdinr_chg_pct", 0.0),
            "nasdaq_chg_pct": _f("nasdaq_chg_pct", 0.0),
            "is_expiry_num": 1.0 if bool(record.get("is_expiry_day", False)) else 0.0,
            "archetype": str(record.get("archetype", "MOMENTUM_MOVER")),
            "entry_session": str(record.get("entry_session", "MORNING_TREND (09:45-11:30)")),
        }

    def log_setup_observation(self, setup_id: str, source_type: str, features_dict: dict):
        """Appends or updates a feature observation keyed by setup_id (trade_id or shadow key)."""
        if not setup_id:
            return
        encoded = self._encode_features(features_dict)
        now_iso = datetime.now(IST).isoformat()

        for row in self.logs:
            if row.get("setup_id") == setup_id:
                row["features"].update(encoded)
                row["updated_iso"] = now_iso
                self._save_logs()
                return

        self.logs.append({
            "setup_id": setup_id,
            "source_type": source_type,  # "REAL_TRADE" or "SHADOW_TRADE"
            "logged_iso": now_iso,
            "label": None,
            "sample_weight": 1.0 if source_type == "REAL_TRADE" else 0.35,
            "pnl_pct": 0.0,
        })
        self.logs[-1]["features"] = encoded
        self._save_logs()

    def sync_labels(self, real_ledger: list, shadow_ledger: list):
        """Attaches binary win/loss labels (1 = profitable net of costs, 0 = loss) from settled setups."""
        label_map = {}

        for t in (real_ledger or []):
            if str(t.get("status", "")).upper() == "OPEN":
                continue
            tid = t.get("trade_id") or f"{t.get('date')}_{t.get('symbol')}"
            if tid:
                pnl_inr = float(t.get("pnl_inr", 0.0))
                label_map[tid] = {
                    "label": 1 if pnl_inr > 0 else 0,
                    "sample_weight": 1.0,
                    "pnl_pct": float(t.get("pnl_pct", 0.0)),
                }

        for s in (shadow_ledger or []):
            st = str(s.get("status", "")).upper()
            if st in ("SHADOW_WIN", "SHADOW_LOSS"):
                sid = f"SHADOW_{s.get('date')}_{s.get('symbol')}_{s.get('entry_iso', '')}"
                label_map[sid] = {
                    "label": 1 if st == "SHADOW_WIN" else 0,
                    "sample_weight": 0.35,
                    "pnl_pct": float(s.get("shadow_pnl_pct", 0.0)),
                }

        updated = 0
        for row in self.logs:
            info = label_map.get(row.get("setup_id"))
            if info and row.get("label") != info["label"]:
                row["label"] = info["label"]
                row["sample_weight"] = info["sample_weight"]
                row["pnl_pct"] = info["pnl_pct"]
                updated += 1

        if updated > 0:
            self._save_logs()
        return updated

    def _build_design_matrix(self, rows: list, cat_maps: dict = None, fit_cats: bool = False):
        if fit_cats or not cat_maps:
            cat_maps = {}
            for col in CATEGORICAL_FEATURES:
                uniq = sorted({str(r.get("features", {}).get(col, "UNKNOWN")) for r in rows})
                cat_maps[col] = {val: float(idx) for idx, val in enumerate(uniq)}

        matrix = []
        for r in rows:
            feat = r.get("features", {})
            vec = [float(feat.get(col, 0.0)) for col in NUMERIC_FEATURES]
            for col in CATEGORICAL_FEATURES:
                c_val = str(feat.get(col, "UNKNOWN"))
                vec.append(cat_maps.get(col, {}).get(c_val, -1.0))
            matrix.append(vec)

        return np.array(matrix, dtype=float), cat_maps

    @staticmethod
    def _make_estimator():
        if HAS_LGBM:
            return LGBMClassifier(
                n_estimators=60,
                learning_rate=0.05,
                num_leaves=7,
                subsample_for_bin=200,
                min_child_samples=8,
                random_state=42,
                verbose=-1,
            ), "LightGBM"
        if HAS_SKLEARN:
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=0.5, max_iter=500, random_state=42)),
            ]), "Sklearn-LogReg"
        return None, "NONE"

    def train_walk_forward(self, real_ledger: list = None, shadow_ledger: list = None) -> dict:
        """
        Trains strictly on chronologically ordered labeled observations using expanding-window
        walk-forward validation so out-of-sample performance is measured without lookahead bias.
        """
        if real_ledger is not None or shadow_ledger is not None:
            self.sync_labels(real_ledger or [], shadow_ledger or [])

        labeled = [r for r in self.logs if r.get("label") in (0, 1)]
        labeled.sort(key=lambda r: str(r.get("logged_iso", "")))

        if len(labeled) < MIN_LABELED_SAMPLES or not (HAS_LGBM or HAS_SKLEARN):
            return {
                "trained": False,
                "samples_count": len(labeled),
                "min_required": MIN_LABELED_SAMPLES,
                "wf_auc": self.artifact.get("wf_auc", 0.50),
            }

        X_all, cat_maps = self._build_design_matrix(labeled, fit_cats=True)
        y_all = np.array([int(r["label"]) for r in labeled], dtype=int)
        w_all = np.array([float(r.get("sample_weight", 1.0)) for r in labeled], dtype=float)

        # Expanding walk-forward evaluation across 4 chronological validation folds
        n_samples = len(labeled)
        val_size = max(5, n_samples // 6)
        oof_probs = []
        oof_trues = []

        start_val = max(25, n_samples - (4 * val_size))
        for cut in range(start_val, n_samples, val_size):
            end_val = min(n_samples, cut + val_size)
            if end_val <= cut:
                continue
            X_tr, y_tr, w_tr = X_all[:cut], y_all[:cut], w_all[:cut]
            X_va, y_va = X_all[cut:end_val], y_all[cut:end_val]
            if len(np.unique(y_tr)) < 2:
                continue

            fold_est, _ = self._make_estimator()
            try:
                fold_est.fit(X_tr, y_tr, sample_weight=w_tr)
            except TypeError:
                fold_est.fit(X_tr, y_tr)

            probs = fold_est.predict_proba(X_va)[:, 1]
            oof_probs.extend(probs.tolist())
            oof_trues.extend(y_va.tolist())

        wf_auc = 0.50
        if len(set(oof_trues)) >= 2 and HAS_SKLEARN:
            try:
                wf_auc = round(float(roc_auc_score(oof_trues, oof_probs)), 3)
            except Exception:
                wf_auc = 0.50

        wf_win_rate = round((sum(oof_trues) / len(oof_trues)) * 100.0, 1) if oof_trues else round(float(y_all.mean()) * 100.0, 1)

        final_est, backend_name = self._make_estimator()
        try:
            final_est.fit(X_all, y_all, sample_weight=w_all)
        except TypeError:
            final_est.fit(X_all, y_all)

        self.artifact = {
            "trained": True,
            "samples_count": n_samples,
            "wf_auc": wf_auc,
            "wf_win_rate": wf_win_rate,
            "backend": backend_name,
            "updated_at": self._now_ist_str(),
            "model": final_est,
            "cat_maps": cat_maps,
        }
        self._save_artifact()
        return {
            "trained": True,
            "samples_count": n_samples,
            "wf_auc": wf_auc,
            "wf_win_rate": wf_win_rate,
            "backend": backend_name,
        }

    def predict_setup_edge(self, candidate_dict: dict) -> dict:
        """
        Returns ML win probability, bounded conviction adjustment (-5.0 to +6.0),
        and status tag for live trade scoring.
        """
        labeled_n = sum(1 for r in self.logs if r.get("label") in (0, 1))
        model = self.artifact.get("model")

        if not self.artifact.get("trained") or model is None or labeled_n < MIN_LABELED_SAMPLES:
            return {
                "ml_active": False,
                "ml_prob": 0.50,
                "ml_conv_adj": 0.0,
                "ml_status": f"Collecting samples ({labeled_n}/{MIN_LABELED_SAMPLES})",
            }

        wrapped = [{"features": self._encode_features(candidate_dict)}]
        X_cand, _ = self._build_design_matrix(wrapped, cat_maps=self.artifact.get("cat_maps", {}), fit_cats=False)

        try:
            prob = float(model.predict_proba(X_cand)[0, 1])
            prob = max(0.15, min(0.85, prob))
            # Center around 0.50; if walk-forward AUC is below 0.52, attenuate adjustment
            auc_scale = 1.0 if float(self.artifact.get("wf_auc", 0.50)) >= 0.52 else 0.4
            raw_adj = (prob - 0.50) * 24.0 * auc_scale
            conv_adj = round(max(-5.0, min(6.0, raw_adj)), 1)
            return {
                "ml_active": True,
                "ml_prob": round(prob, 4),
                "ml_conv_adj": conv_adj,
                "ml_status": f"{self.artifact.get('backend', 'ML')} ({prob:.0%}, AUC {self.artifact.get('wf_auc', 0.5):.2f})",
            }
        except Exception:
            return {
                "ml_active": False,
                "ml_prob": 0.50,
                "ml_conv_adj": 0.0,
                "ml_status": "ML fallback neutral",
            }