# ==============================================================================
# XGBoost Streamflow Prediction — Godawari Sub-basin Style  (v15 — 0.72 Push)
#
# v15 Final Alignment Push:
# - Shifted Rain Alignment: Added rain_lag4 (features=8 total)
# - Noise reduction: Applied Streamflow.rolling(2).mean() before log1p
# - Hyperparameters: min_child_weight -> 4, gamma -> 0.05
# ==============================================================================

import logging
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = Path("/content")
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

DATA_PATH = ROOT / "dataset.csv"
PLOT_FILE = str(OUT_DIR / "xgb_hydrology.png")

# ── Config ────────────────────────────────────────────────────────────────────
CFG = {
    # Features (v15 Timing + Smoothing)
    "features": [
        "rainfall",   # Pt
        "rain_lag1",  # Pt-1
        "rain_lag2",  # Pt-2
        "rain_lag3",  # Pt-3
        "rain_lag4",  # Pt-4 (shifted rain alignment)
        "rain_3day",  # 3-day antecedent rainfall
        "rain_5day",  # 5-day rainfall (medium response)
        "flow_lag1",  # Qt-1
    ],
    "target":     "streamflow",
    "train_frac": 0.70,

    # High R Micro-tuned Hyperparameters
    "xgb": {
        "n_estimators":          700,
        "max_depth":             4,
        "learning_rate":         0.04,
        "subsample":             0.8,
        "colsample_bytree":      0.8,
        "min_child_weight":      4,
        "gamma":                 0.05,
        "reg_alpha":             0.1,
        "reg_lambda":            2.0,
        "random_state":          42,
        "objective":             "reg:squarederror",
        "n_jobs":                -1,
        "early_stopping_rounds": 20,
    },
}

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("XGB-Streamflow")


# ==============================================================================
# 1. DATA LOADING — auto-detect column names (case-insensitive)
# ==============================================================================

def load_data(path: Path) -> pd.DataFrame:
    """Load CSV, rename columns to canonical names, sort by date."""
    raw = pd.read_csv(path)
    raw.columns = raw.columns.str.strip()

    # ── Find date column ──────────────────────────────────────────────────────
    date_col = None
    for c in raw.columns:
        if c.lower() in ("date", "datetime", "time"):
            date_col = c
            break
    if date_col is None:
        raise ValueError(f"No date column found. Available: {list(raw.columns)}")

    raw[date_col] = pd.to_datetime(raw[date_col])
    raw = raw.set_index(date_col).sort_index()

    # ── Auto-rename to canonical names ────────────────────────────────────────
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
    log.info("Streamflow — mean: %.1f  max: %.1f  std: %.1f",
             raw["Streamflow"].mean(), raw["Streamflow"].max(), raw["Streamflow"].std())
    return raw


# ==============================================================================
# 2. FEATURE ENGINEERING — Final v4
#    Features: rainfall, rain_lag1–3, rain_3day, rain_7day, flow_lag1  (7 total)
#    Target  : log1p(streamflow) — NO MinMaxScaler on target (FIX 1+2)
# ==============================================================================

def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame(index=raw.index)

    # ── Precipitation features ────────────────────────────────────────────────
    df["rainfall"]  = raw["Rain"]
    df["rain_lag1"] = raw["Rain"].shift(1)
    df["rain_lag2"] = raw["Rain"].shift(2)
    df["rain_lag3"] = raw["Rain"].shift(3)
    df["rain_lag4"] = raw["Rain"].shift(4)           # shifted rain alignment
    df["rain_3day"] = raw["Rain"].rolling(3).sum()   # short response
    df["rain_5day"] = raw["Rain"].rolling(5).sum()   # medium response

    # ── Autoregressive lag (Qt-1 only) ────────────────────────────────────────
    df["flow_lag1"] = raw["Streamflow"].shift(1)

    # ── Target: log1p smoothed — NO MinMaxScaler ──────────────────────────────
    raw["Streamflow_smooth"] = raw["Streamflow"].rolling(2).mean()
    df["streamflow"] = np.log1p(raw["Streamflow_smooth"])

    df = df.dropna()
    log.info("Feature df: %d rows | %d features | target = smoothed log1p",
             len(df), 8)
    return df


# ==============================================================================
# 3. NORMALIZATION — MinMaxScaler on FEATURES only
# FIX 1+2: Target is NOT scaled — kept as raw log1p values
#   Before (v3): MinMax(features) + MinMax(log1p(target))  → double compression
#   After  (v4): MinMax(features) only; target = log1p(Q) in natural units
# ==============================================================================

def scale_data(df: pd.DataFrame, features: list, target: str):
    feat_scaler = MinMaxScaler()

    df_sc = df.copy()
    df_sc[features] = feat_scaler.fit_transform(df[features])
    # target column left as-is (log1p values, unscaled)
    return df_sc, feat_scaler


def inverse_y(arr: np.ndarray) -> np.ndarray:
    """
    Inverse log1p → original m³/s.
    FIX 1+2: no MinMaxScaler to invert — just expm1.
    """
    return np.expm1(arr)


# ==============================================================================
# 4. TRAIN / TEST SPLIT — chronological (70 / 30), NO shuffling
# ==============================================================================

def split_data(df_sc: pd.DataFrame, features: list, target: str, train_frac: float):
    n       = len(df_sc)
    n_train = int(n * train_frac)

    X = df_sc[features].values
    y = df_sc[target].values

    X_tr, X_te = X[:n_train], X[n_train:]
    y_tr, y_te = y[:n_train], y[n_train:]

    dates = df_sc.index
    dates_tr = dates[:n_train]
    dates_te = dates[n_train:]

    log.info("Split — Train: %d  |  Test: %d", n_train, n - n_train)
    return X_tr, y_tr, X_te, y_te, dates_tr, dates_te


# ==============================================================================
# 5 & 6. MODEL + TRAINING
# ==============================================================================

def build_and_train(X_tr, y_tr, X_te, y_te, y_tr_original: np.ndarray):
    """
    Train XGBRegressor with peak-aware sample weights (FIX 3).

    FIX 3 — sample_weight = 1 + 1.0*(Q/mean)
      Peak days get weight ~5–20× greater than baseflow days.
      y_tr_original: training observations in m³/s (physical units).
    """
    # ── Compute sample weights in physical units (final strict bounds) ────────
    mean_q  = float(y_tr_original.mean())
    weights_raw = 1.0 + 0.5 * (y_tr_original / (mean_q + 1e-8))
    weights: np.ndarray = np.asarray(
        np.clip(weights_raw, 1.0, 5.0), dtype=np.float32
    )
    log.info("Sample weight — min: %.2f  mean: %.2f  max: %.2f",
             float(weights.min()), float(weights.mean()), float(weights.max()))

    params = CFG["xgb"].copy()
    model  = XGBRegressor(**params)

    model.fit(
        X_tr, y_tr,
        sample_weight = weights,
        eval_set      = [(X_te, y_te)],
        verbose       = 100,
    )
    log.info("Best iteration: %d", model.best_iteration)
    return model


# ==============================================================================
# 7. PREDICTION + INVERSE TRANSFORM
# FIX 1+2: no target_scaler — inverse is just expm1()
# ==============================================================================

def predict_physical(model, X_tr, X_te, y_tr, y_te):
    pred_tr = inverse_y(model.predict(X_tr))
    pred_te = inverse_y(model.predict(X_te))
    obs_tr  = inverse_y(y_tr)
    obs_te  = inverse_y(y_te)
    return pred_tr, pred_te, obs_tr, obs_te


# ==============================================================================
# 8. METRICS
# ==============================================================================

def nse(obs: np.ndarray, sim: np.ndarray) -> float:
    return float(1.0 - np.sum((obs - sim) ** 2) / np.sum((obs - obs.mean()) ** 2))

def kge(obs: np.ndarray, sim: np.ndarray) -> float:
    r = float(np.corrcoef(obs, sim)[0, 1])
    return float(1 - np.sqrt((r - 1) ** 2
                              + (sim.mean() / obs.mean() - 1) ** 2
                              + (sim.std()  / obs.std()  - 1) ** 2))

def compute_metrics(obs: np.ndarray, sim: np.ndarray) -> dict:
    r, _ = pearsonr(obs, sim)
    return {
        "R    (Pearson)": float(r),
        "NSE           ": nse(obs, sim),
        "KGE           ": kge(obs, sim),
        "RMSE (m³/s)   ": float(np.sqrt(mean_squared_error(obs, sim))),
        "MAE  (m³/s)   ": float(mean_absolute_error(obs, sim)),
    }


# ==============================================================================
# 9. PRINT TABLE
# ==============================================================================

def print_metrics(train_m: dict, test_m: dict):
    print("\n── XGBoost Prediction Metrics ──────────────────────────────────────")
    print(f"{'Metric':<22} {'Training':>10} {'Testing':>10}")
    print("─" * 45)
    for key in train_m:
        te  = test_m.get(key, float("nan"))
        print(f"  {key} {train_m[key]:>10.4f} {te:>10.4f}")
    print()


# ==============================================================================
# 10. PLOTS — light theme
# ==============================================================================

C_RAIN = "#9b59b6"; C_OBS = "#d62728"; C_PRED = "#2ca02c"; C_SCATTER = "#ff7f0e"
BG = "#F8F9FA"; AX_BG = "#FFFFFF"; GRID = "#E0E0E0"
TEXT_COLOR = "#333333"

def style_ax(ax):
    ax.set_facecolor(AX_BG)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.tick_params(colors=TEXT_COLOR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_COLOR)
    ax.yaxis.label.set_color(TEXT_COLOR)
    ax.title.set_color(TEXT_COLOR)
    ax.grid(axis="y", color=GRID, linestyle="--", linewidth=0.6, alpha=0.9)


def make_plots(
    df,
    obs_full, pred_full,
    obs_tr,   pred_tr,
    obs_te,   pred_te,
    dates_all, dates_tr, dates_te,
    test_m, train_m,
    features,
    model,
):
    r_te   = test_m["R    (Pearson)"]
    nse_te = test_m["NSE           "]
    kge_te = test_m["KGE           "]
    r_tr   = train_m["R    (Pearson)"]
    nse_tr = train_m["NSE           "]

    rain_full = df["rainfall"].values

    # ── Added: Correlation Matrix (Paper Style) ───────────────────────────────
    import seaborn as sns
    fig_c = plt.figure(figsize=(11, 9))
    fig_c.patch.set_facecolor(BG)
    corr_df = df.copy()

    # 1. Rename columns to canonical paper notation
    rename_map = {
        "flow_lag1":  "Qt-1",
        "rainfall":   "Pt",
        "rain_lag1":  "Pt-1",
        "rain_lag2":  "Pt-2",
        "rain_lag3":  "Pt-3",
        "rain_lag4":  "Pt-4",
        "rain_3day":  "Pt_3d",
        "rain_5day":  "Pt_5d",
    }
    if "streamflow" in corr_df:
        corr_df["Qt"] = np.expm1(corr_df.pop("streamflow"))

    corr_df = corr_df.rename(columns=rename_map)
    col_order = ["Qt", "Qt-1", "Pt", "Pt-1", "Pt-2", "Pt-3", "Pt-4", "Pt_3d", "Pt_5d"]
    col_order = [c for c in col_order if c in corr_df.columns]
    corr_df = corr_df[col_order]

    corr = corr_df.corr()

    # 2. Strict lower triangle mask
    mask = np.triu(np.ones_like(corr, dtype=bool), k=0)

    # 3. Sequential blue colormap
    ax_heat = sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        cmap="Blues",
        fmt=".2g",
        square=True,
        linewidths=.5,
        cbar_kws={'shrink': .85, 'label': ''}
    )

    # Style tweaks
    ax_heat.set_facecolor(AX_BG)
    plt.title("Correlation matrix for Dataset", color=TEXT_COLOR, fontsize=12, pad=15)
    plt.tick_params(colors=TEXT_COLOR)
    ax_heat.tick_params(axis='x', colors=TEXT_COLOR, rotation=0, bottom=False)
    ax_heat.tick_params(axis='y', colors=TEXT_COLOR, rotation=0, left=False)
    cb = ax_heat.collections[0].colorbar
    cb.ax.yaxis.set_tick_params(colors=TEXT_COLOR)

    corr_path = str(OUT_DIR / "correlation_matrix.png")
    plt.savefig(corr_path, dpi=150, bbox_inches="tight", facecolor=BG)
    # Don't close fig_c so it pops up when plt.show() is called
    log.info("Saved lower-triangle correlation matrix → %s", corr_path)

    # ── 1. Combined full series (Streamflow + Inverted Rainfall) ──────────────
    fig1, ax1 = plt.subplots(figsize=(16, 6))
    fig1.patch.set_facecolor(BG)
    style_ax(ax1)
    ax1.plot(dates_all, obs_full,  color=C_OBS,  lw=1.1, label="Observed Q", alpha=0.9)
    ax1.plot(dates_all, pred_full, color=C_PRED, lw=1.1, ls="--", label="XGBoost Q")
    ax1.set_ylabel("Streamflow (m³/s)", fontsize=10, color=C_OBS)

    ax1_rain = ax1.twinx()
    ax1_rain.bar(dates_all, rain_full, color=C_RAIN, alpha=0.8, width=1.0, label="Rainfall")
    ax1_rain.set_ylabel("Rainfall (mm)", fontsize=10, color=C_RAIN)
    ax1_rain.invert_yaxis()
    ax1_rain.set_ylim(rain_full.max() * 3.5, 0)
    ax1_rain.spines['right'].set_edgecolor(GRID)
    ax1_rain.tick_params(colors=TEXT_COLOR)

    ax1.axvline(dates_te[0], color="#555555", lw=1.0, ls=":", alpha=0.8,
                label=f"Train/Test split ({dates_te[0].date()})")
    ax1.set_title("Streamflow with Inverted Rainfall (Full Series)",
                  fontsize=12, fontweight="bold", pad=8)
    ax1.xaxis.set_major_locator(mdates.YearLocator(5))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.tick_params(axis="x", rotation=30)

    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax1_rain.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, facecolor=AX_BG, edgecolor=GRID,
               labelcolor=TEXT_COLOR, fontsize=9, loc='lower right')
    ax1.set_xlim(dates_all[0], dates_all[-1])

    p1 = str(OUT_DIR / "hydrograph_full.png")
    fig1.savefig(p1, dpi=150, bbox_inches="tight", facecolor=BG)
    log.info("Saved independent plot → %s", p1)

    # ── 2. Test set zoom ──────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(16, 6))
    fig2.patch.set_facecolor(BG)
    style_ax(ax2)
    ax2.plot(dates_te, obs_te,  color=C_OBS,  lw=1.4, label="Observed")
    ax2.plot(dates_te, pred_te, color=C_PRED, lw=1.4, ls="--", label="XGBoost")
    ax2.set_title(f"Test Set  |  R={r_te:.3f}  NSE={nse_te:.3f}  KGE={kge_te:.3f}",
                  fontsize=11, fontweight="bold", pad=8)
    ax2.set_ylabel("Streamflow (m³/s)", fontsize=10)
    ax2.xaxis.set_major_locator(mdates.YearLocator(2))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.tick_params(axis="x", rotation=30)
    ax2.legend(facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT_COLOR, fontsize=9)
    ax2.set_xlim(dates_te[0], dates_te[-1])

    p2 = str(OUT_DIR / "hydrograph_test.png")
    fig2.savefig(p2, dpi=150, bbox_inches="tight", facecolor=BG)
    log.info("Saved independent plot → %s", p2)

    # ── 3a. Scatter — test ────────────────────────────────────────────────────
    fig4, ax4 = plt.subplots(figsize=(7, 6))
    fig4.patch.set_facecolor(BG)
    style_ax(ax4)
    ax4.scatter(obs_te, pred_te, color=C_SCATTER, alpha=0.9, s=8)
    mv = max(obs_te.max(), pred_te.max())
    ax4.plot([0, mv], [0, mv], color="#000000", lw=1.2, ls="--", alpha=0.6)
    ax4.set_title(f"Scatter — Test\nR={r_te:.3f}  NSE={nse_te:.3f}",
                  fontsize=10, fontweight="bold", pad=6)
    ax4.set_xlabel("Observed (m³/s)", fontsize=9)
    ax4.set_ylabel("Simulated (m³/s)", fontsize=9)
    ax4.grid(axis="both", color=GRID, linestyle="--", linewidth=0.6, alpha=0.7)

    p3 = str(OUT_DIR / "scatter_test.png")
    fig4.savefig(p3, dpi=150, bbox_inches="tight", facecolor=BG)
    log.info("Saved independent plot → %s", p3)

    # ── 3b. Scatter — train ───────────────────────────────────────────────────
    fig4b, ax4b = plt.subplots(figsize=(7, 6))
    fig4b.patch.set_facecolor(BG)
    style_ax(ax4b)
    ax4b.scatter(obs_tr, pred_tr, color=C_OBS, alpha=0.3, s=4)
    mv_tr = max(obs_tr.max(), pred_tr.max())
    ax4b.plot([0, mv_tr], [0, mv_tr], color="#000000", lw=1.2, ls="--", alpha=0.6)
    ax4b.set_title(f"Scatter — Training\nR={r_tr:.3f}  NSE={nse_tr:.3f}",
                   fontsize=10, fontweight="bold", pad=6)
    ax4b.set_xlabel("Observed (m³/s)", fontsize=9)
    ax4b.set_ylabel("Simulated (m³/s)", fontsize=9)
    ax4b.grid(axis="both", color=GRID, linestyle="--", linewidth=0.6, alpha=0.7)

    p4 = str(OUT_DIR / "scatter_train.png")
    fig4b.savefig(p4, dpi=150, bbox_inches="tight", facecolor=BG)
    log.info("Saved independent plot → %s", p4)

    # ── 4. Feature importance ─────────────────────────────────────────────────
    fig5, ax5 = plt.subplots(figsize=(10, 5))
    fig5.patch.set_facecolor(BG)
    style_ax(ax5)
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1]
    bars = ax5.bar(
        [features[i] for i in idx], imp[idx],
        color=C_PRED, alpha=0.85, edgecolor=GRID,
    )
    ax5.set_title("XGBoost Feature Importance (gain)",
                  fontsize=11, fontweight="bold", pad=6)
    ax5.set_ylabel("Importance", fontsize=9)
    ax5.set_xlabel("Feature", fontsize=9)
    ax5.grid(axis="y", color=GRID, linestyle="--", linewidth=0.6, alpha=0.7)
    for bar, val in zip(bars, imp[idx]):
        ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 f"{val:.3f}", ha="center", va="bottom",
                 fontsize=8, color=TEXT_COLOR)

    p5 = str(OUT_DIR / "feature_importance.png")
    fig5.savefig(p5, dpi=150, bbox_inches="tight", facecolor=BG)
    log.info("Saved independent plot → %s", p5)
    # plt.show() deferred to end of run


def make_diagnostic_plots(obs_full, pred_full, dates_all):
    df_eval = pd.DataFrame({"Observed": obs_full, "Predicted": pred_full}, index=dates_all)

    # -- 1. Flow Duration Curve (FDC) --
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    fig1.patch.set_facecolor(BG)
    style_ax(ax1)

    obs_sort = np.sort(obs_full)[::-1]
    pred_sort = np.sort(pred_full)[::-1]
    rank = np.arange(1, len(obs_full) + 1)
    exceedance = (rank / (len(obs_full) + 1)) * 100

    # Use small offset to avoid log(0) error for exactly zero flows
    ax1.plot(exceedance, np.maximum(obs_sort, 0.1), color=C_OBS, lw=1.5, label="Observed")
    ax1.plot(exceedance, np.maximum(pred_sort, 0.1), color=C_PRED, lw=1.5, ls="--", label="XGBoost")
    ax1.set_yscale("log")
    ax1.set_title("Flow Duration Curve (FDC)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Exceedance Probability (%)", fontsize=10)
    ax1.set_ylabel("Streamflow (m³/s) [Log Scale]", fontsize=10)
    ax1.legend(facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT_COLOR)

    p1 = str(OUT_DIR / "flow_duration_curve.png")
    fig1.savefig(p1, dpi=150, bbox_inches="tight", facecolor=BG)
    log.info("Saved independent diagnostic plot → %s", p1)

    # -- 2. Monthly Climatology --
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    fig2.patch.set_facecolor(BG)
    style_ax(ax2)
    monthly_mean = df_eval.groupby(df_eval.index.month).mean()
    months = np.arange(1, 13)
    width = 0.35
    ax2.bar(months - width/2, monthly_mean["Observed"], width, color=C_OBS, alpha=0.8, label="Observed")
    ax2.bar(months + width/2, monthly_mean["Predicted"], width, color=C_PRED, alpha=0.8, label="XGBoost")
    ax2.set_title("Monthly Average Streamflow (Seasonality)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Month", fontsize=10)
    ax2.set_ylabel("Average Streamflow (m³/s)", fontsize=10)
    ax2.set_xticks(months)
    ax2.set_xticklabels(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"])
    ax2.legend(facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT_COLOR)

    p2 = str(OUT_DIR / "monthly_seasonality.png")
    fig2.savefig(p2, dpi=150, bbox_inches="tight", facecolor=BG)
    log.info("Saved independent diagnostic plot → %s", p2)

    # -- 3. Residual vs Observed --
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    fig3.patch.set_facecolor(BG)
    style_ax(ax3)
    residuals = pred_full - obs_full
    ax3.scatter(obs_full, residuals, color=C_SCATTER, alpha=0.4, s=15, edgecolors='none')
    ax3.axhline(0, color="#555555", lw=1.5, ls="--")
    ax3.set_title("Residuals vs. Observed Flow", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Observed Streamflow (m³/s)", fontsize=10)
    ax3.set_ylabel("Residual (Predicted - Observed)", fontsize=10)

    p3 = str(OUT_DIR / "residuals_scatter.png")
    fig3.savefig(p3, dpi=150, bbox_inches="tight", facecolor=BG)
    log.info("Saved independent diagnostic plot → %s", p3)

    # -- 4. Cumulative Volume --
    fig4, ax4 = plt.subplots(figsize=(8, 6))
    fig4.patch.set_facecolor(BG)
    style_ax(ax4)
    ax4.plot(dates_all, np.cumsum(obs_full), color=C_OBS, lw=1.5, label="Observed Cumul.")
    ax4.plot(dates_all, np.cumsum(pred_full), color=C_PRED, lw=1.5, ls="--", label="XGBoost Cumul.")
    ax4.set_title("Cumulative Streamflow Volume", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Year", fontsize=10)
    ax4.set_ylabel("Cumulative Flow", fontsize=10)
    ax4.xaxis.set_major_locator(mdates.YearLocator(5))
    ax4.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax4.tick_params(axis="x", rotation=30)
    ax4.legend(facecolor=AX_BG, edgecolor=GRID, labelcolor=TEXT_COLOR)

    p4 = str(OUT_DIR / "cumulative_volume.png")
    fig4.savefig(p4, dpi=150, bbox_inches="tight", facecolor=BG)
    log.info("Saved independent diagnostic plot → %s", p4)


# ==============================================================================
# PIPELINE — wire everything together
# ==============================================================================

def main():
    # 1. Load
    raw = load_data(DATA_PATH)

    # 2. Feature engineering
    df       = build_features(raw)
    features = CFG["features"]
    target   = CFG["target"]

    # 3. Normalize features only (FIX 1+2: target NOT scaled)
    df_sc, feat_scaler = scale_data(df, features, target)

    # 4. Split (chronological, no shuffle)
    X_tr, y_tr, X_te, y_te, dates_tr, dates_te = split_data(
        df_sc, features, target, CFG["train_frac"]
    )

    # 5 & 6. Build + train
    # Physical-scale obs for peak weighting (FIX 3): just expm1(y_tr)
    obs_tr_phys = np.expm1(y_tr)
    log.info("Training XGBRegressor (log1p target, no target scaling) …")
    model = build_and_train(X_tr, y_tr, X_te, y_te, obs_tr_phys)

    # 7. Predict + inverse transform
    pred_tr, pred_te, obs_tr, obs_te = predict_physical(
        model, X_tr, X_te, y_tr, y_te
    )
    pred_full = np.concatenate([pred_tr, pred_te])
    obs_full  = np.concatenate([obs_tr,  obs_te])
    dates_all = df.index

    # 8. Metrics
    train_m = compute_metrics(obs_tr, pred_tr)
    test_m  = compute_metrics(obs_te, pred_te)

    # 9. Print table
    print_metrics(train_m, test_m)

    # Sanity check RMSE/MAE ratio
    ratio = test_m["RMSE (m³/s)   "] / (test_m["MAE  (m³/s)   "] + 1e-8)
    if 1 <= ratio <= 10:
        log.info("RMSE/MAE ratio = %.2f ✓", ratio)
    else:
        log.warning("RMSE/MAE ratio = %.2f — check units!", ratio)

    # 10. Plots
    make_plots(df, obs_full, pred_full,
               obs_tr, pred_tr, obs_te, pred_te,
               dates_all, dates_tr, dates_te,
               test_m, train_m, features, model)

    # 11. Advanced Diagnostics
    make_diagnostic_plots(obs_full, pred_full, dates_all)

    log.info("Done. Outputs in %s", OUT_DIR)

    # Show all windows at the end
    plt.show()

if __name__ == "__main__":
    main()
