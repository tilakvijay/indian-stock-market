from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    data_provider: Literal["auto", "jugaad", "yfinance"] = Field(default="auto", alias="DATA_PROVIDER")
    default_exchange: Literal["NSE", "BSE"] = Field(default="NSE", alias="DEFAULT_EXCHANGE")
    history_period: str = Field(default="5y", alias="HISTORY_PERIOD")
    bar_interval: str = Field(default="1d", alias="BAR_INTERVAL")
    intraday_interval: str = Field(default="5m", alias="INTRADAY_INTERVAL")
    intraday_lookback_days: int = Field(default=30, alias="INTRADAY_LOOKBACK_DAYS")
    data_cache_dir: str = Field(default=".cache/market_data", alias="DATA_CACHE_DIR")
    jugaad_cache_dir: str = Field(default=".cache/jugaad", alias="JUGAAD_CACHE_DIR")
    screener_universe_size: int = Field(default=20, alias="SCREENER_UNIVERSE_SIZE")

    # PatchTST deep model settings (optimized for RTX 3050 4GB)
    enable_torch_model: bool = Field(default=False, alias="ENABLE_TORCH_MODEL")
    torch_epochs: int = Field(default=50, alias="TORCH_EPOCHS")
    torch_sequence_length: int = Field(default=64, alias="TORCH_SEQUENCE_LENGTH")
    torch_batch_size: int = Field(default=32, alias="TORCH_BATCH_SIZE")
    patchtst_patch_len: int = Field(default=16, alias="PATCHTST_PATCH_LEN")
    patchtst_stride: int = Field(default=8, alias="PATCHTST_STRIDE")
    patchtst_d_model: int = Field(default=64, alias="PATCHTST_D_MODEL")
    patchtst_n_heads: int = Field(default=4, alias="PATCHTST_N_HEADS")
    patchtst_n_layers: int = Field(default=3, alias="PATCHTST_N_LAYERS")
    patchtst_dropout: float = Field(default=0.1, alias="PATCHTST_DROPOUT")

    # Regime detection
    regime_persistence_days: int = Field(default=3, alias="REGIME_PERSISTENCE_DAYS")

    # Position sizing
    kelly_fraction_cap: float = Field(default=0.25, alias="KELLY_FRACTION_CAP")

    # Walk-forward backtesting
    wf_folds: int = Field(default=5, alias="WF_FOLDS")
    wf_purge_bars: int = Field(default=10, alias="WF_PURGE_BARS")
    wf_embargo_bars: int = Field(default=5, alias="WF_EMBARGO_BARS")
    monte_carlo_simulations: int = Field(default=1000, alias="MONTE_CARLO_SIMULATIONS")

    request_timeout_seconds: int = 15


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
