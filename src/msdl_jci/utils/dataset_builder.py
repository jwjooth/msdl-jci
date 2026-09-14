"""Multi-Source Dataset Builder and Preprocessor for MSDL-JCI.

Implements the exact methodology described in Sections 2.1 - 2.3 of the thesis:
1. Technical feature engineering (RSI-14, MACD + Signal, ATR-14, SMA-20, Normalized Close & Volume)
2. Macroeconomic data point-in-time forward-fill alignment (BI-Rate, Inflation, Kurs USD/IDR)
3. Financial news semantic embedding alignment (768-dim Frozen IndoBERT, weekend rollover, zero-vector imputation)
4. Temporal alignment and t+5 directional movement labeling (UP=1, DOWN=0)
5. Zero-leakage sliding window tensor generation (Lookback=28)
"""

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from msdl_jci.config.settings import get_settings
from msdl_jci.utils.logging_config import get_logger

logger = get_logger(__name__)

MONTH_MAP = {
    "januari": "01", "jan": "01",
    "februari": "02", "feb": "02",
    "maret": "03", "mar": "03",
    "april": "04", "apr": "04",
    "mei": "05", "may": "05",
    "juni": "06", "jun": "06",
    "juli": "07", "jul": "07",
    "agustus": "08", "agu": "08", "aug": "08",
    "september": "09", "sep": "09",
    "oktober": "10", "okt": "10", "oct": "10",
    "november": "11", "nov": "11",
    "desember": "12", "des": "12", "dec": "12",
}


@dataclass
class MultiSourceTensors:
    """Container for multi-source aligned PyTorch-ready tensors."""

    # 3D Technical sequence: [Samples, Lookback (28), Num_Features]
    X_tech: np.ndarray
    # 2D Macro features: [Samples, 3] (BI-Rate, Inflation, USD/IDR)
    X_macro: np.ndarray
    # 2D News embeddings: [Samples, 768] (Frozen IndoBERT)
    X_news: np.ndarray
    # 1D Directional labels: [Samples] (1 for UP, 0 for DOWN over t+5)
    y: np.ndarray
    # 1D Forward 5-day returns: [Samples] ((Close_{t+5} - Close_t) / Close_t)
    returns_5d: np.ndarray
    # Aligned trade dates: [Samples]
    dates: np.ndarray
    # Actual close prices at time t: [Samples]
    close_prices: np.ndarray
    # Feature column names for interpretability
    tech_feature_cols: list[str]


def parse_indonesian_date(series: pd.Series, day_first: bool = True) -> pd.Series:
    """Parse Indonesian text date strings into pandas datetime series."""
    s = series.astype(str).str.strip().str.lower()
    for id_month, num_month in MONTH_MAP.items():
        s = s.str.replace(id_month, num_month, regex=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return pd.to_datetime(s, errors="coerce", dayfirst=day_first)


def calculate_technical_indicators(
    df: pd.DataFrame,
    rsi_period: int = 14,
    atr_period: int = 14,
    sma_period: int = 20,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
) -> pd.DataFrame:
    """Calculate technical momentum, trend, and volatility indicators.

    Features computed (matching Thesis Section 2.2):
    - RSI-14: Relative Strength Index with Wilder's smoothing
    - MACD Line & Signal Line: 12/26/9 EMA divergence
    - ATR-14: Average True Range volatility measure
    - SMA-20: 20-day Simple Moving Average trend context
    - Close & Volume: Raw levels for normalization
    """
    df = df.copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values("Date").reset_index(drop=True)

    # 1. RSI-14 (Wilder's exponential smoothing)
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / rsi_period, min_periods=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / rsi_period, min_periods=rsi_period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    df["RSI_14"] = 100.0 - (100.0 / (1.0 + rs))

    # 2. MACD (12, 26, 9)
    ema_fast = df["Close"].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=macd_slow, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=macd_signal, adjust=False).mean()

    # 3. ATR-14
    prev_close = df["Close"].shift(1)
    tr1 = df["High"] - df["Low"]
    tr2 = (df["High"] - prev_close).abs()
    tr3 = (df["Low"] - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["ATR_14"] = tr.ewm(alpha=1.0 / atr_period, min_periods=atr_period, adjust=False).mean()

    # 4. SMA-20
    df["SMA_20"] = df["Close"].rolling(window=sma_period).mean()

    return df


class MultiSourceDatasetBuilder:
    """Builds unified multi-source dataset from technical, macro, and news sources."""

    def __init__(
        self,
        look_back: int | None = None,
        prediction_horizon: int | None = None,
    ) -> None:
        settings = get_settings()
        self.look_back = look_back if look_back is not None else settings.ML_LOOK_BACK
        self.prediction_horizon = (
            prediction_horizon
            if prediction_horizon is not None
            else settings.ML_PREDICTION_HORIZON
        )
        self.tech_feature_cols = [
            "Close",
            "Volume",
            "RSI_14",
            "MACD",
            "MACD_Signal",
            "ATR_14",
            "SMA_20",
        ]

    def load_raw_macro_data(
        self,
        bi_rate_path: str | Path,
        inflation_path: str | Path,
        kurs_path: str | Path,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and parse raw macroeconomic data tables."""
        # 1. BI Rate
        df_bi = pd.read_csv(bi_rate_path)
        bi_col = "BI-7Day-RR" if "BI-7Day-RR" in df_bi.columns else df_bi.columns[-1]
        period_col = "Period" if "Period" in df_bi.columns else df_bi.columns[1]
        df_bi["Date"] = parse_indonesian_date(df_bi[period_col])
        df_bi["bi_rate"] = (
            df_bi[bi_col]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.strip()
            .astype(float)
        )
        df_bi = df_bi.dropna(subset=["Date"]).sort_values("Date")[["Date", "bi_rate"]].copy()

        # 2. Inflation Data
        df_inf = pd.read_csv(inflation_path)
        inf_col = "Data Inflasi" if "Data Inflasi" in df_inf.columns else df_inf.columns[-1]
        inf_pcol = "Periode" if "Periode" in df_inf.columns else df_inf.columns[0]
        df_inf["Date"] = parse_indonesian_date(df_inf[inf_pcol], day_first=False)
        df_inf["inflation_rate"] = (
            df_inf[inf_col]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.strip()
            .astype(float)
        )
        df_inf = df_inf.dropna(subset=["Date"]).sort_values("Date")[["Date", "inflation_rate"]].copy()

        # 3. Kurs USD/IDR
        df_kurs = pd.read_csv(kurs_path)
        df_kurs["Date"] = pd.to_datetime(df_kurs["Date"], errors="coerce")
        df_kurs["usd_idr"] = pd.to_numeric(
            df_kurs["Close"].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )
        df_kurs = df_kurs.dropna(subset=["Date"]).sort_values("Date")[["Date", "usd_idr"]].copy()

        return df_bi, df_inf, df_kurs

    def build_aligned_dataframe(
        self,
        jci_csv: str | Path | None = None,
        bi_rate_csv: str | Path | None = None,
        inflation_csv: str | Path | None = None,
        kurs_csv: str | Path | None = None,
        news_emb_csv: str | Path | None = None,
    ) -> pd.DataFrame:
        """Merge all modalities into a synchronized master dataframe."""
        settings = get_settings()
        jci_path = jci_csv or settings.JCI_HISTORICAL_CSV
        bi_path = bi_rate_csv or settings.BI_RATE_CSV
        inf_path = inflation_csv or settings.INFLATION_CSV
        kurs_path = kurs_csv or settings.KURS_CSV
        news_path = news_emb_csv or settings.DAILY_NEWS_EMBEDDINGS_CSV

        # 1. Technical Data (JCI OHLCV)
        logger.info("Loading JCI technical data from %s", jci_path)
        df_jci = pd.read_csv(jci_path)
        df_jci["Date"] = pd.to_datetime(df_jci["Date"], errors="coerce")
        df_jci = df_jci.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
        df_tech = calculate_technical_indicators(df_jci)

        # 2. Macroeconomic Data Alignment (Forward-fill LOCF)
        logger.info("Aligning macroeconomic data via forward-fill...")
        df_bi, df_inf, df_kurs = self.load_raw_macro_data(bi_path, inf_path, kurs_path)

        # Merge onto trading dates calendar
        df_master = pd.merge(df_tech, df_kurs, on="Date", how="left")
        df_master = pd.merge_asof(
            df_master.sort_values("Date"),
            df_bi.sort_values("Date"),
            on="Date",
            direction="backward",
        )
        df_master = pd.merge_asof(
            df_master.sort_values("Date"),
            df_inf.sort_values("Date"),
            on="Date",
            direction="backward",
        )

        # Forward-fill periodic policy values.
        # NOTE (leakage fix): NO backfill. bfill() would inject future macro
        # values into the start of the dataset. Rows with genuinely
        # unavailable macro data are dropped instead.
        df_master["usd_idr"] = df_master["usd_idr"].ffill()
        df_master["bi_rate"] = df_master["bi_rate"].ffill()
        df_master["inflation_rate"] = df_master["inflation_rate"].ffill()
        # Drop leading rows where macro data was not yet available (no look-ahead).
        df_master = df_master.dropna(subset=["usd_idr", "bi_rate", "inflation_rate"]).reset_index(drop=True)

        # 3. Target Labeling for Horizon t+5 (Thesis Section 2.3)
        h = self.prediction_horizon
        future_close = df_master["Close"].shift(-h)
        return_5d = (future_close - df_master["Close"]) / df_master["Close"]
        target_dir = (future_close > df_master["Close"]).astype(float)

        df_master["future_close"] = future_close
        df_master["return_5d"] = return_5d
        df_master["target_direction"] = target_dir

        # 4. Financial News Embeddings Alignment (768-dim Frozen IndoBERT)
        if Path(news_path).exists():
            logger.info("Loading precomputed IndoBERT embeddings from %s", news_path)
            df_news = pd.read_csv(news_path)
            df_news = df_news.rename(columns={"trade_date": "Date"})
            df_news["Date"] = pd.to_datetime(df_news["Date"], format="%Y-%m-%d", errors="coerce")
            df_news = df_news.copy()
            emb_cols = [c for c in df_news.columns if c.startswith("emb_")]

            df_master = pd.merge(df_master, df_news, on="Date", how="left")
            # Thesis Section 2.3: Zero-vector representation for days without news
            df_master[emb_cols] = df_master[emb_cols].fillna(0.0)
        else:
            logger.warning("News embeddings file %s not found. Creating zero vectors.", news_path)
            emb_cols = [f"emb_{i}" for i in range(settings.NEWS_EMB_DIM)]
            zeros_df = pd.DataFrame(0.0, index=df_master.index, columns=emb_cols)
            df_master = pd.concat([df_master, zeros_df], axis=1)

        # Clean NaN rows caused by technical indicator lookback and future target shift
        df_master = df_master.dropna(subset=self.tech_feature_cols).reset_index(drop=True)
        # Separate the labeled valid rows (where future_close is known)
        df_labeled = df_master.dropna(subset=["future_close"]).copy().reset_index(drop=True)

        logger.info(
            "Aligned dataset ready: %d samples (total dates: %s to %s)",
            len(df_labeled),
            df_labeled["Date"].min().strftime("%Y-%m-%d"),
            df_labeled["Date"].max().strftime("%Y-%m-%d"),
        )
        return df_labeled

    def create_multisource_tensors(
        self,
        df: pd.DataFrame,
        feature_scaler: MinMaxScaler | None = None,
        macro_scaler: MinMaxScaler | None = None,
        fit_scalers: bool = True,
        train_end_idx: int | None = None,
    ) -> tuple[MultiSourceTensors, MinMaxScaler, MinMaxScaler]:
        """Convert aligned dataframe into sliding window sequential and point-in-time tensors.

        Args:
            df: Aligned master dataframe (chronological, from build_aligned_dataframe).
            feature_scaler: Pre-fitted scaler for technical features (required if fit_scalers=False).
            macro_scaler: Pre-fitted scaler for macro features (required if fit_scalers=False).
            fit_scalers: If True, fit new scalers (on train prefix if train_end_idx given,
                else on full df — legacy behaviour, NOT recommended for evaluation).
            train_end_idx: Optional row index (in ``df`` space) delimiting the training
                prefix. When provided with fit_scalers=True, scalers are fitted ONLY on
                ``df.iloc[:train_end_idx]`` rows and then applied to the full frame.
                This prevents test-period feature ranges from leaking into training.

        Returns:
            (tensors, feature_scaler, macro_scaler)
        """
        # --- Leakage guard: future_close must never become a feature ---
        forbidden = {"future_close", "return_5d", "target_direction"}
        leaked = forbidden.intersection(set(self.tech_feature_cols))
        if leaked:
            raise ValueError(f"tech_feature_cols must not contain target columns: {leaked}")
        for col in ("future_close", "return_5d", "target_direction"):
            if col in self.tech_feature_cols:
                raise ValueError(f"Leakage: '{col}' must never be included in features")

        macro_cols = ["bi_rate", "inflation_rate", "usd_idr"]
        emb_cols = [c for c in df.columns if c.startswith("emb_")]

        raw_tech = df[self.tech_feature_cols].values.astype(np.float32)
        raw_macro = df[macro_cols].values.astype(np.float32)
        raw_news = df[emb_cols].values.astype(np.float32)
        raw_targets = df["target_direction"].values.astype(np.float32)
        raw_returns = df["return_5d"].values.astype(np.float32)
        raw_dates = df["Date"].dt.strftime("%Y-%m-%d").values
        raw_close = df["Close"].values.astype(np.float32)

        # Scaling — fit on TRAIN PREFIX ONLY when train_end_idx is given.
        if fit_scalers:
            feature_scaler = MinMaxScaler(feature_range=(0, 1))
            macro_scaler = MinMaxScaler(feature_range=(0, 1))
            if train_end_idx is not None:
                if not (0 < train_end_idx <= len(df)):
                    raise ValueError(
                        f"train_end_idx={train_end_idx} out of range for df len={len(df)}"
                    )
                # Fit strictly on rows that precede the validation/test boundary.
                feature_scaler.fit(raw_tech[:train_end_idx])
                macro_scaler.fit(raw_macro[:train_end_idx])
                logger.info(
                    "Fitted scalers on train prefix [: %d] / %d rows (leak-free)",
                    train_end_idx,
                    len(df),
                )
            else:
                logger.warning(
                    "Fitting scalers on FULL dataframe (%d rows) — leaks test ranges. "
                    "Pass train_end_idx for leak-free evaluation.",
                    len(df),
                )
                feature_scaler.fit(raw_tech)
                macro_scaler.fit(raw_macro)
            scaled_tech = feature_scaler.transform(raw_tech)
            scaled_macro = macro_scaler.transform(raw_macro)
        else:
            if feature_scaler is None or macro_scaler is None:
                raise ValueError("Must provide fitted scalers when fit_scalers=False")
            scaled_tech = feature_scaler.transform(raw_tech)
            scaled_macro = macro_scaler.transform(raw_macro)

        # Sliding window construction: [Samples, Lookback, D_tech]
        lb = self.look_back
        n_samples = len(df) - lb + 1
        if n_samples <= 0:
            raise ValueError(f"Dataset has {len(df)} rows, less than look_back={lb}")

        X_tech_list = []
        X_macro_list = []
        X_news_list = []
        y_list = []
        returns_list = []
        dates_list = []
        close_list = []

        for i in range(lb - 1, len(df)):
            window = scaled_tech[i - lb + 1 : i + 1]
            X_tech_list.append(window)
            X_macro_list.append(scaled_macro[i])
            X_news_list.append(raw_news[i])
            y_list.append(raw_targets[i])
            returns_list.append(raw_returns[i])
            dates_list.append(raw_dates[i])
            close_list.append(raw_close[i])

        tensors = MultiSourceTensors(
            X_tech=np.array(X_tech_list, dtype=np.float32),
            X_macro=np.array(X_macro_list, dtype=np.float32),
            X_news=np.array(X_news_list, dtype=np.float32),
            y=np.array(y_list, dtype=np.float32),
            returns_5d=np.array(returns_list, dtype=np.float32),
            dates=np.array(dates_list),
            close_prices=np.array(close_list, dtype=np.float32),
            tech_feature_cols=self.tech_feature_cols,
        )

        return tensors, feature_scaler, macro_scaler
