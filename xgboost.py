# ==============================================================================
# Multi-Model Streamflow Prediction — Godawari Sub-basin Style
#
# Models: XGBoost | AdaBoost | CatBoost | LightGBM
# ==============================================================================

import logging
import subprocess
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# ── Auto-install CatBoost ─────────────────────────────────────────────────────
try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    print("CatBoost not found — installing now …")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "catboost", "-q"])
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
    print("CatBoost installed successfully.")

# ── Auto-install LightGBM ─────────────────────────────────────────────────────
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    print("LightGBM not found — installing now …")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "lightgbm", "-q"])
    import lightgbm as lgb
    HAS_LGB = True
    print("LightGBM installed successfully.")

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path("/content")
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

DATA_PATH = ROOT / "dataset.csv"

# ── Config ────────────────────────────────────────────────────────────────────
FEATURES = [
    "rainfall",   # Pt
    "rain_lag1",  # Pt-1
    "rain_lag2",  # Pt-2
    "rain_lag3",  # Pt-3
    "rain_lag4",  # Pt-4
    "rain_3day",  # 3-day antecedent rainfall
    "rain_5day",  # 5-day antecedent rainfall
    "flow_lag1",  # Qt-1
]
TARGET      = "streamflow"
TRAIN_FRAC  = 0.70

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("MultiModel-Streamflow")

# ── Plot constants ─────────────────────────────────────────────────────────────
BG         = "#F8F9FA"
AX_BG      = "#FFFFFF"
GRID       = "#E0E0E0"
TEXT_COLOR = "#333333"

MODEL_COLORS = {
    "XGBoost":  "#2ca02c",
    "AdaBoost": "#1f77b4",
    "CatBoost": "#ff7f0e",
    "LightGBM": "#9467bd",
}
C_OBS  = "#d62728"
C_RAIN = "#9b59b6"


# ==============================================================================
# 1. DATA LOADING
# ==============================================================================

def load_data(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw.columns = raw.columns.str.strip()

    date_col = None
    for c in raw.columns:
        if c.lower() in ("date", "datetime", "time"):
            date_col = c
            break
    if date_col is None:
        raise ValueError(f"No date column found. Available: {list(raw.columns)}")

    raw[date_col] = pd.to_datetime(raw[date_col])
    raw = raw.set_index(date_col).sort_index()

    rename_map = {}
    for c in raw.columns:
        cl = c.lower()
        if any(x in cl for x in ("rain", "prec", "precip", "rf", "p_")):
            rename_map[c] = "Rain"
        elif any(x in cl for x in ("flow", "discharge", "runoff", "streamflow", "q_")):
            rename_map[c] = "Streamflow"

    raw = raw.rename(columns=rename_map)
    log.info("Columns after rename: %s", list(raw.columns))

    if "Rain" not in raw.columns:
        raise ValueError(f"Rainfall column not found. Have: {list(raw.columns)}")
    if "Streamflow" not in raw.columns:
        raise ValueError(f"Streamflow column not found. Have: {list(raw.columns)}")

    log.info("Loaded %d rows  |  %s → %s",
             len(raw), raw.index[0].date(), raw.index[-1].date())
    return raw


# ==============================================================================
# 2. FEATURE ENGINEERING
# ==============================================================================

def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=raw.index)
    df["rainfall"]  = raw["Rain"]
    df["rain_lag1"] = raw["Rain"].shift(1)
    df["rain_lag2"] = raw["Rain"].shift(2)
    df["rain_lag3"] = raw["Rain"].shift(3)
    df["rain_lag4"] = raw["Rain"].shift(4)
    df["rain_3day"] = raw["Rain"].rolling(3).sum()
    df["rain_5day"] = raw["Rain"].rolling(5).sum()
    df["flow_lag1"] = raw["Streamflow"].shift(1)

    raw["Streamflow_smooth"] = raw["Streamflow"].rolling(2).mean()
    df["streamflow"] = np.log1p(raw["Streamflow_smooth"])

    df = df.dropna()
    log.info("Feature df: %d rows | %d features", len(df), len(FEATURES))
    return df


# ==============================================================================
# 3. NORMALISE (features only) + SPLIT
# ==============================================================================

def prepare_data(df: pd.DataFrame):
    scaler = MinMaxScaler()
    df_sc = df.copy()
    df_sc[FEATURES] = scaler.fit_transform(df[FEATURES])

    n       = len(df_sc)
    n_train = int(n * TRAIN_FRAC)

    X = df_sc[FEATURES].values
    y = df_sc[TARGET].values

    X_tr, X_te = X[:n_train], X[n_train:]
    y_tr, y_te = y[:n_train], y[n_train:]
    dates      = df_sc.index
    dates_tr   = dates[:n_train]
    dates_te   = dates[n_train:]

    log.info("Split — Train: %d  |  Test: %d", n_train, n - n_train)
    return X_tr, y_tr, X_te, y_te, dates_tr, dates_te, scaler


# ==============================================================================
# 4. MODEL DEFINITIONS
# ==============================================================================

def get_models() -> dict:
    models = {}

    models["XGBoost"] = XGBRegressor(
        n_estimators          = 700,
        max_depth             = 4,
        learning_rate         = 0.04,
        subsample             = 0.8,
        colsample_bytree      = 0.8,
        min_child_weight      = 4,
        gamma                 = 0.05,
        reg_alpha             = 0.1,
        reg_lambda            = 2.0,
        random_state          = 42,
        objective             = "reg:squarederror",
        n_jobs                = -1,
        early_stopping_rounds = 20,
    )

    base_tree = DecisionTreeRegressor(max_depth=4, random_state=42)
    models["AdaBoost"] = AdaBoostRegressor(
        estimator    = base_tree,
        n_estimators = 500,
        learning_rate= 0.05,
        loss         = "linear",
        random_state = 42,
    )

    if HAS_CATBOOST:
        models["CatBoost"] = CatBoostRegressor(
            iterations          = 700,
            depth               = 4,
            learning_rate       = 0.04,
            l2_leaf_reg         = 3.0,
            random_seed         = 42,
            verbose             = 100,
            allow_writing_files = False,
        )

    if HAS_LGB:
        models["LightGBM"] = lgb.LGBMRegressor(
            n_estimators     = 700,
            max_depth        = 4,
            learning_rate    = 0.04,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            min_child_weight = 4,
            reg_alpha        = 0.1,
            reg_lambda       = 2.0,
            random_state     = 42,
            n_jobs           = -1,
            verbose          = -1,
        )

    return models


# ==============================================================================
# 5. TRAINING — with peak-aware sample weights
# ==============================================================================

def train_model(name: str, model, X_tr, y_tr, X_te, y_te):
    obs_tr_phys = np.expm1(y_tr)
    mean_q      = float(obs_tr_phys.mean())
    weights     = np.clip(
        1.0 + 0.5 * (obs_tr_phys / (mean_q + 1e-8)), 1.0, 5.0
    ).astype(np.float32)

    log.info("Training %s …", name)

    if name == "XGBoost":
        model.fit(
            X_tr, y_tr,
            sample_weight = weights,
            eval_set      = [(X_te, y_te)],
            verbose       = 100,
        )
    elif name == "AdaBoost":
        model.fit(X_tr, y_tr, sample_weight=weights)
    elif name == "CatBoost":
        model.fit(
            X_tr, y_tr,
            sample_weight  = weights,
            eval_set       = (X_te, y_te),
            use_best_model = True,
        )
    elif name == "LightGBM":
        callbacks = [lgb.early_stopping(20, verbose=False),
                     lgb.log_evaluation(100)]
        model.fit(
            X_tr, y_tr,
            sample_weight = weights,
            eval_set      = [(X_te, y_te)],
            callbacks     = callbacks,
        )
    else:
        model.fit(X_tr, y_tr)

    log.info("%s training complete.", name)
    return model


# ==============================================================================
# 6. METRICS
# ==============================================================================

def nse(obs, sim):
    return float(1.0 - np.sum((obs - sim) ** 2) / np.sum((obs - obs.mean()) ** 2))

def kge(obs, sim):
    r = float(np.corrcoef(obs, sim)[0, 1])
    return float(1 - np.sqrt((r - 1) ** 2
                              + (sim.mean() / obs.mean() - 1) ** 2
                              + (sim.std()  / obs.std()  - 1) ** 2))

def compute_metrics(obs, sim):
    r, _ = pearsonr(obs, sim)
    return {
        "R":    float(r),
        "NSE":  nse(obs, sim),
        "KGE":  kge(obs, sim),
        "RMSE": float(np.sqrt(mean_squared_error(obs, sim))),
        "MAE":  float(mean_absolute_error(obs, sim)),
    }


# ==============================================================================
# 7. PLOT HELPERS
# ==============================================================================

def style_ax(ax):
    ax.set_facecolor(AX_BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    ax.grid(axis="y", color=GRID, linestyle="--", linewidth=0.6, alpha=0.9)


def save_close(fig, path: str):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    log.info("Saved → %s", path)


# ==============================================================================
# 8. PER-MODEL PLOTS
# ==============================================================================

def plot_per_model(name, obs_tr, pred_tr, obs_te, pred_te,
                   obs_full, pred_full,
                   dates_tr, dates_te, dates_all,
                   raw_rain_mm,
                   train_m, test_m, model):

    color = MODEL_COLORS.get(name, "#333333")
    slug  = name.lower().replace(" ", "_")

    # ── A. Hydrograph — Test period ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(16, 5))
    fig.patch.set_facecolor(BG)
    style_ax(ax)
    ax.plot(dates_te, obs_te,  color=C_OBS, lw=1.4, label="Observed")
    ax.plot(dates_te, pred_te, color=color, lw=1.4, ls="--", label=name)
    ax.set_title(
        f"{name} — Test Set  |  R={test_m['R']:.3f}  "
        f"NSE={test_m['NSE']:.3f}  KGE={test_m['KGE']:.3f}",
        fontsize=11, fontweight="bold", pad=8)
    ax.set_ylabel("Streamflow (m³/s)", fontsize=10)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.tick_params(axis="x", rotation=30)
    ax.legend(facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT_COLOR, fontsize=9)
    ax.set_xlim(dates_te[0], dates_te[-1])
    save_close(fig, str(OUT_DIR / f"{slug}_hydrograph_test.png"))

    # ── B. Full series with inverted rainfall ────────────────────────────────
    rain_full = raw_rain_mm.reindex(dates_all).values
    fig2, ax2 = plt.subplots(figsize=(16, 5))
    fig2.patch.set_facecolor(BG)
    style_ax(ax2)
    ax2.plot(dates_all, obs_full,  color=C_OBS, lw=1.0, label="Observed Q", alpha=0.9)
    ax2.plot(dates_all, pred_full, color=color, lw=1.0, ls="--", label=f"{name} Q")
    ax2.set_ylabel("Streamflow (m³/s)", fontsize=10, color=C_OBS)

    ax2r = ax2.twinx()
    ax2r.bar(dates_all, rain_full, color=C_RAIN, alpha=0.7, width=1.0, label="Rainfall")
    ax2r.set_ylabel("Rainfall (mm)", fontsize=10, color=C_RAIN)
    ax2r.invert_yaxis()
    ax2r.set_ylim(np.nanmax(rain_full) * 3.5, 0)
    ax2r.spines['right'].set_edgecolor(GRID)
    ax2r.tick_params(colors=TEXT_COLOR)

    ax2.axvline(dates_te[0], color="#555555", lw=1.0, ls=":",
                label=f"Train/Test split ({dates_te[0].date()})")
    ax2.set_title(f"{name} — Full Series with Rainfall", fontsize=11, fontweight="bold", pad=8)
    ax2.xaxis.set_major_locator(mdates.YearLocator(5))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.tick_params(axis="x", rotation=30)
    l1, lb1 = ax2.get_legend_handles_labels()
    l2, lb2 = ax2r.get_legend_handles_labels()
    ax2.legend(l1 + l2, lb1 + lb2, facecolor=AX_BG, edgecolor=GRID,
               labelcolor=TEXT_COLOR, fontsize=9, loc="lower right")
    ax2.set_xlim(dates_all[0], dates_all[-1])
    save_close(fig2, str(OUT_DIR / f"{slug}_hydrograph_full.png"))

    # ── C. Scatter — Test ────────────────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    fig3.patch.set_facecolor(BG)
    style_ax(ax3)
    ax3.scatter(obs_te, pred_te, color=color, alpha=0.6, s=8)
    mv = max(obs_te.max(), pred_te.max())
    ax3.plot([0, mv], [0, mv], color="#000", lw=1.2, ls="--", alpha=0.6)
    ax3.set_title(f"{name} — Scatter (Test)\nR={test_m['R']:.3f}  NSE={test_m['NSE']:.3f}",
                  fontsize=10, fontweight="bold", pad=6)
    ax3.set_xlabel("Observed (m³/s)", fontsize=9)
    ax3.set_ylabel("Simulated (m³/s)", fontsize=9)
    ax3.grid(axis="both", color=GRID, linestyle="--", linewidth=0.6, alpha=0.7)
    save_close(fig3, str(OUT_DIR / f"{slug}_scatter_test.png"))

    # ── D. Scatter — Train ───────────────────────────────────────────────────
    fig3b, ax3b = plt.subplots(figsize=(6, 6))
    fig3b.patch.set_facecolor(BG)
    style_ax(ax3b)
    ax3b.scatter(obs_tr, pred_tr, color=color, alpha=0.3, s=4)
    mv_tr = max(obs_tr.max(), pred_tr.max())
    ax3b.plot([0, mv_tr], [0, mv_tr], color="#000", lw=1.2, ls="--", alpha=0.6)
    ax3b.set_title(f"{name} — Scatter (Training)\nR={train_m['R']:.3f}  NSE={train_m['NSE']:.3f}",
                   fontsize=10, fontweight="bold", pad=6)
    ax3b.set_xlabel("Observed (m³/s)", fontsize=9)
    ax3b.set_ylabel("Simulated (m³/s)", fontsize=9)
    ax3b.grid(axis="both", color=GRID, linestyle="--", linewidth=0.6, alpha=0.7)
    save_close(fig3b, str(OUT_DIR / f"{slug}_scatter_train.png"))

    # ── E. Feature Importance (if available) ─────────────────────────────────
    imp = None
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "get_feature_importance"):
        imp = model.get_feature_importance()

    if imp is not None:
        fig4, ax4 = plt.subplots(figsize=(10, 4))
        fig4.patch.set_facecolor(BG)
        style_ax(ax4)
        idx  = np.argsort(imp)[::-1]
        bars = ax4.bar([FEATURES[i] for i in idx], imp[idx],
                       color=color, alpha=0.85, edgecolor=GRID)
        ax4.set_title(f"{name} — Feature Importance", fontsize=11, fontweight="bold", pad=6)
        ax4.set_ylabel("Importance", fontsize=9)
        ax4.set_xlabel("Feature", fontsize=9)
        for bar, val in zip(bars, imp[idx]):
            ax4.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + imp.max() * 0.01,
                     f"{val:.3f}", ha="center", va="bottom",
                     fontsize=8, color=TEXT_COLOR)
        save_close(fig4, str(OUT_DIR / f"{slug}_feature_importance.png"))


# ==============================================================================
# 9. PER-MODEL DIAGNOSTIC PLOTS
#    Flow Duration Curve | Monthly Seasonality | Residuals | Cumulative Volume
# ==============================================================================

def plot_diagnostics_per_model(name, obs_full, pred_full, dates_all):
    slug  = name.lower().replace(" ", "_")
    color = MODEL_COLORS.get(name, "#333333")

    df_eval = pd.DataFrame(
        {"Observed": obs_full, "Predicted": pred_full}, index=dates_all
    )

    # ── 1. Flow Duration Curve ────────────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    fig1.patch.set_facecolor(BG)
    style_ax(ax1)
    obs_sort  = np.sort(obs_full)[::-1]
    pred_sort = np.sort(pred_full)[::-1]
    rank      = np.arange(1, len(obs_full) + 1)
    exceed    = (rank / (len(obs_full) + 1)) * 100
    ax1.plot(exceed, np.maximum(obs_sort,  0.1), color=C_OBS,  lw=1.5, label="Observed")
    ax1.plot(exceed, np.maximum(pred_sort, 0.1), color=color,  lw=1.5, ls="--", label=name)
    ax1.set_yscale("log")
    ax1.set_title(f"{name} — Flow Duration Curve", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Exceedance Probability (%)", fontsize=10)
    ax1.set_ylabel("Streamflow (m³/s) [Log Scale]", fontsize=10)
    ax1.legend(facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT_COLOR)
    save_close(fig1, str(OUT_DIR / f"{slug}_flow_duration_curve.png"))

    # ── 2. Monthly Climatology ────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    fig2.patch.set_facecolor(BG)
    style_ax(ax2)
    monthly = df_eval.groupby(df_eval.index.month).mean()
    months  = np.arange(1, 13)
    width   = 0.35
    ax2.bar(months - width/2, monthly["Observed"],  width, color=C_OBS,  alpha=0.8, label="Observed")
    ax2.bar(months + width/2, monthly["Predicted"], width, color=color,  alpha=0.8, label=name)
    ax2.set_title(f"{name} — Monthly Average Streamflow", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Month", fontsize=10)
    ax2.set_ylabel("Average Streamflow (m³/s)", fontsize=10)
    ax2.set_xticks(months)
    ax2.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun",
                          "Jul","Aug","Sep","Oct","Nov","Dec"])
    ax2.legend(facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT_COLOR)
    save_close(fig2, str(OUT_DIR / f"{slug}_monthly_seasonality.png"))

    # ── 3. Residuals vs Observed ──────────────────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    fig3.patch.set_facecolor(BG)
    style_ax(ax3)
    residuals = pred_full - obs_full
    ax3.scatter(obs_full, residuals, color=color, alpha=0.4, s=12, edgecolors="none")
    ax3.axhline(0, color="#555555", lw=1.5, ls="--")
    ax3.set_title(f"{name} — Residuals vs Observed", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Observed Streamflow (m³/s)", fontsize=10)
    ax3.set_ylabel("Residual (Predicted − Observed)", fontsize=10)
    ax3.grid(axis="both", color=GRID, linestyle="--", linewidth=0.6, alpha=0.7)
    save_close(fig3, str(OUT_DIR / f"{slug}_residuals.png"))

    # ── 4. Cumulative Volume ──────────────────────────────────────────────────
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    fig4.patch.set_facecolor(BG)
    style_ax(ax4)
    ax4.plot(dates_all, np.cumsum(obs_full),  color=C_OBS, lw=1.5, label="Observed Cumul.")
    ax4.plot(dates_all, np.cumsum(pred_full), color=color, lw=1.5, ls="--", label=f"{name} Cumul.")
    ax4.set_title(f"{name} — Cumulative Streamflow Volume", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Year", fontsize=10)
    ax4.set_ylabel("Cumulative Flow (m³/s·day)", fontsize=10)
    ax4.xaxis.set_major_locator(mdates.YearLocator(5))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax4.tick_params(axis="x", rotation=30)
    ax4.legend(facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT_COLOR)
    save_close(fig4, str(OUT_DIR / f"{slug}_cumulative_volume.png"))


# ==============================================================================
# 10. COMPARISON PLOTS
# ==============================================================================

def plot_comparison(all_results: dict, dates_te, obs_te):

    # ── A. Metric bar chart comparison (R, NSE, KGE) ─────────────────────────
    metrics_order = ["R", "NSE", "KGE"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor(BG)
    fig.suptitle("Model Comparison — Test Set Metrics",
                 fontsize=13, fontweight="bold", color=TEXT_COLOR, y=1.01)
    for ax, metric in zip(axes, metrics_order):
        style_ax(ax)
        names  = list(all_results.keys())
        values = [all_results[n]["test"][metric] for n in names]
        colors = [MODEL_COLORS.get(n, "#888") for n in names]
        bars   = ax.bar(names, values, color=colors, alpha=0.85, edgecolor=GRID, width=0.5)
        ax.set_title(metric, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.05)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=9, color=TEXT_COLOR)
        ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    save_close(fig, str(OUT_DIR / "comparison_metrics_bar.png"))

    # ── B. RMSE & MAE comparison ──────────────────────────────────────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(11, 5))
    fig2.patch.set_facecolor(BG)
    fig2.suptitle("Model Comparison — Error Metrics (Test Set)",
                  fontsize=13, fontweight="bold", color=TEXT_COLOR, y=1.01)
    for ax, metric in zip(axes2, ["RMSE", "MAE"]):
        style_ax(ax)
        names  = list(all_results.keys())
        values = [all_results[n]["test"][metric] for n in names]
        colors = [MODEL_COLORS.get(n, "#888") for n in names]
        bars   = ax.bar(names, values, color=colors, alpha=0.85, edgecolor=GRID, width=0.5)
        ax.set_title(f"{metric} (m³/s)", fontsize=11, fontweight="bold")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.01,
                    f"{val:.2f}", ha="center", va="bottom",
                    fontsize=9, color=TEXT_COLOR)
        ax.tick_params(axis="x", rotation=15)
    plt.tight_layout()
    save_close(fig2, str(OUT_DIR / "comparison_error_bar.png"))

    # ── C. Overlay hydrograph — test period ──────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(18, 6))
    fig3.patch.set_facecolor(BG)
    style_ax(ax3)
    ax3.plot(dates_te, obs_te, color=C_OBS, lw=1.6, label="Observed", zorder=5)
    for name, res in all_results.items():
        ax3.plot(dates_te, res["pred_te"],
                 color=MODEL_COLORS.get(name, "#888"), lw=1.2, ls="--",
                 label=f"{name}  R={res['test']['R']:.3f}  NSE={res['test']['NSE']:.3f}")
    ax3.set_title("All Models — Test Period Hydrograph Overlay",
                  fontsize=12, fontweight="bold", pad=8)
    ax3.set_ylabel("Streamflow (m³/s)", fontsize=10)
    ax3.xaxis.set_major_locator(mdates.YearLocator(2))
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.tick_params(axis="x", rotation=30)
    ax3.legend(facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT_COLOR,
               fontsize=9, loc="upper left")
    ax3.set_xlim(dates_te[0], dates_te[-1])
    save_close(fig3, str(OUT_DIR / "comparison_hydrograph_overlay.png"))

    # ── D. Scatter 2×2 grid ───────────────────────────────────────────────────
    names = list(all_results.keys())
    ncols = 2
    nrows = (len(names) + 1) // 2
    fig4, axes4 = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows))
    fig4.patch.set_facecolor(BG)
    fig4.suptitle("Scatter Plots — Test Set (All Models)",
                  fontsize=13, fontweight="bold", color=TEXT_COLOR)
    axes4_flat = axes4.flatten() if hasattr(axes4, "flatten") else [axes4]
    for ax, name in zip(axes4_flat, names):
        style_ax(ax)
        pred  = all_results[name]["pred_te"]
        color = MODEL_COLORS.get(name, "#888")
        ax.scatter(obs_te, pred, color=color, alpha=0.5, s=8)
        mv = max(obs_te.max(), pred.max())
        ax.plot([0, mv], [0, mv], color="#000", lw=1.2, ls="--", alpha=0.6)
        m = all_results[name]["test"]
        ax.set_title(f"{name}\nR={m['R']:.3f}  NSE={m['NSE']:.3f}  KGE={m['KGE']:.3f}",
                     fontsize=10, fontweight="bold", pad=5)
        ax.set_xlabel("Observed (m³/s)", fontsize=9)
        ax.set_ylabel("Simulated (m³/s)", fontsize=9)
        ax.grid(axis="both", color=GRID, linestyle="--", linewidth=0.6, alpha=0.7)
    for ax in axes4_flat[len(names):]:
        ax.set_visible(False)
    plt.tight_layout()
    save_close(fig4, str(OUT_DIR / "comparison_scatter_grid.png"))

    # ── E. FDC overlay — all models ───────────────────────────────────────────
    fig5, ax5 = plt.subplots(figsize=(10, 7))
    fig5.patch.set_facecolor(BG)
    style_ax(ax5)
    n = len(obs_te)
    rank   = np.arange(1, n + 1)
    exceed = (rank / (n + 1)) * 100
    obs_sort = np.sort(obs_te)[::-1]
    ax5.plot(exceed, np.maximum(obs_sort, 0.1), color=C_OBS, lw=2.0, label="Observed", zorder=5)
    for name, res in all_results.items():
        pred_sort = np.sort(res["pred_te"])[::-1]
        ax5.plot(exceed, np.maximum(pred_sort, 0.1),
                 color=MODEL_COLORS.get(name, "#888"), lw=1.4, ls="--", label=name)
    ax5.set_yscale("log")
    ax5.set_title("Flow Duration Curve — All Models (Test Period)",
                  fontsize=12, fontweight="bold", pad=8)
    ax5.set_xlabel("Exceedance Probability (%)", fontsize=10)
    ax5.set_ylabel("Streamflow (m³/s) [Log Scale]", fontsize=10)
    ax5.legend(facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT_COLOR, fontsize=9)
    save_close(fig5, str(OUT_DIR / "comparison_fdc_overlay.png"))

    # ── F. Monthly climatology overlay — all models ───────────────────────────
    fig6, ax6 = plt.subplots(figsize=(12, 6))
    fig6.patch.set_facecolor(BG)
    style_ax(ax6)
    months       = np.arange(1, 13)
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    obs_monthly = pd.Series(obs_te, index=dates_te).groupby(
        pd.Series(obs_te, index=dates_te).index.month).mean()

    n_models    = len(all_results)
    total_width = 0.7
    bar_w       = total_width / (n_models + 1)
    offsets     = np.linspace(-total_width/2, total_width/2, n_models + 1)

    ax6.bar(months + offsets[0], obs_monthly.reindex(months, fill_value=0),
            bar_w, color=C_OBS, alpha=0.85, label="Observed")
    for i, (name, res) in enumerate(all_results.items()):
        pred_monthly = pd.Series(res["pred_te"], index=dates_te).groupby(
            pd.Series(res["pred_te"], index=dates_te).index.month).mean()
        ax6.bar(months + offsets[i + 1], pred_monthly.reindex(months, fill_value=0),
                bar_w, color=MODEL_COLORS.get(name, "#888"), alpha=0.85, label=name)

    ax6.set_title("Monthly Average Streamflow — All Models (Test Period)",
                  fontsize=12, fontweight="bold", pad=8)
    ax6.set_xlabel("Month", fontsize=10)
    ax6.set_ylabel("Average Streamflow (m³/s)", fontsize=10)
    ax6.set_xticks(months)
    ax6.set_xticklabels(month_labels)
    ax6.legend(facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT_COLOR, fontsize=9)
    save_close(fig6, str(OUT_DIR / "comparison_monthly_overlay.png"))


# ==============================================================================
# 11. CORRELATION MATRIX
# ==============================================================================

def plot_correlation_matrix(df: pd.DataFrame):
    corr_df   = df.copy()
    rename_map = {
        "flow_lag1": "Qt-1", "rainfall":  "Pt",    "rain_lag1": "Pt-1",
        "rain_lag2": "Pt-2", "rain_lag3": "Pt-3",  "rain_lag4": "Pt-4",
        "rain_3day": "Pt_3d","rain_5day": "Pt_5d",
    }
    if "streamflow" in corr_df:
        corr_df["Qt"] = np.expm1(corr_df.pop("streamflow"))
    corr_df   = corr_df.rename(columns=rename_map)
    col_order = ["Qt","Qt-1","Pt","Pt-1","Pt-2","Pt-3","Pt-4","Pt_3d","Pt_5d"]
    col_order = [c for c in col_order if c in corr_df.columns]
    corr_df   = corr_df[col_order]
    corr      = corr_df.corr()
    mask      = np.triu(np.ones_like(corr, dtype=bool), k=0)

    fig = plt.figure(figsize=(10, 8))
    fig.patch.set_facecolor(BG)
    ax  = sns.heatmap(corr, mask=mask, annot=True, cmap="Blues",
                      fmt=".2g", square=True, linewidths=.5,
                      cbar_kws={"shrink": .85})
    ax.set_facecolor(AX_BG)
    plt.title("Correlation Matrix for Dataset", color=TEXT_COLOR, fontsize=12, pad=15)
    ax.tick_params(axis="x", colors=TEXT_COLOR, rotation=0, bottom=False)
    ax.tick_params(axis="y", colors=TEXT_COLOR, rotation=0, left=False)
    cb = ax.collections[0].colorbar
    cb.ax.yaxis.set_tick_params(colors=TEXT_COLOR)
    save_close(fig, str(OUT_DIR / "correlation_matrix.png"))


# ==============================================================================
# 12. PRINT COMPARISON TABLE
# ==============================================================================

def print_comparison_table(all_results: dict):
    metrics = ["R", "NSE", "KGE", "RMSE", "MAE"]
    header  = f"{'Model':<12}"
    for phase in ("Train", "Test"):
        for m in metrics:
            header += f"  {phase[:2]}_{m:>4}"
    print("\n── Multi-Model Streamflow Prediction — Results ──────────────────────────────────────────────────────────")
    print(header)
    print("─" * len(header))

    for name, res in all_results.items():
        row = f"{name:<12}"
        for phase_key in ("train", "test"):
            for m in metrics:
                val = res[phase_key][m]
                row += f"  {val:>7.3f}"
        print(row)

    print()
    ranked = sorted(all_results.items(), key=lambda x: x[1]["test"]["NSE"], reverse=True)
    print("  Ranking by Test NSE:")
    for rank, (name, res) in enumerate(ranked, 1):
        print(f"    {rank}. {name:<10}  NSE={res['test']['NSE']:.4f}  "
              f"R={res['test']['R']:.4f}  KGE={res['test']['KGE']:.4f}")
    print()


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    # 1. Load raw data
    raw = load_data(DATA_PATH)

    # 2. Feature engineering
    df = build_features(raw)

    # Keep the actual mm rainfall aligned to the feature-df index
    raw_rain_mm = raw["Rain"].reindex(df.index)

    # 3. Correlation matrix
    plot_correlation_matrix(df)

    # 4. Prepare (normalise + split)
    X_tr, y_tr, X_te, y_te, dates_tr, dates_te, scaler = prepare_data(df)

    # Physical-scale obs arrays
    obs_tr    = np.expm1(y_tr)
    obs_te    = np.expm1(y_te)
    obs_full  = np.concatenate([obs_tr, obs_te])
    dates_all = df.index

    # 5. Get model registry
    models = get_models()

    # 6. Train, predict, evaluate, plot each model
    all_results = {}

    for name, model in models.items():
        model = train_model(name, model, X_tr, y_tr, X_te, y_te)

        pred_tr_phys = np.expm1(model.predict(X_tr))
        pred_te_phys = np.expm1(model.predict(X_te))
        pred_full    = np.concatenate([pred_tr_phys, pred_te_phys])

        train_m = compute_metrics(obs_tr, pred_tr_phys)
        test_m  = compute_metrics(obs_te, pred_te_phys)

        all_results[name] = {
            "train":     train_m,
            "test":      test_m,
            "pred_te":   pred_te_phys,
            "pred_tr":   pred_tr_phys,
            "pred_full": pred_full,
            "model":     model,
        }

        # Per-model main plots
        plot_per_model(
            name,
            obs_tr,       pred_tr_phys,
            obs_te,       pred_te_phys,
            obs_full,     pred_full,
            dates_tr,     dates_te,    dates_all,
            raw_rain_mm,
            train_m,      test_m,      model,
        )

        # Per-model diagnostic plots
        plot_diagnostics_per_model(name, obs_full, pred_full, dates_all)

    # 7. Print comparison table
    print_comparison_table(all_results)

    # 8. Comparison plots
    plot_comparison(all_results, dates_te, obs_te)

    log.info("All done. Outputs saved to %s", OUT_DIR)
    plt.show()


if __name__ == "__main__":
    main()
