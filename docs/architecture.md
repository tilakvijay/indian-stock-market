# Pipeline Map

## Layer 1: Data extraction

- `CompanyResolver`
  - Yahoo Finance search fallback for open access
  - NSE/BSE suffix normalization for direct symbols
- `JugaadDataProvider`
  - Historical daily OHLCV using `jugaad-data`
  - Bhavcopy equity and F&O downloads
- `AlternativeDataCollector`
  - `nsepython` option-chain snapshot
  - `nsepython` FII/DII snapshot
  - `jugaad-data` live quote and bhavcopy context
- `YahooFinanceProvider`
  - Backup daily OHLCV
  - Recent 5 minute bars for realized volatility and cross-validation

## Layer 2: Feature engineering

- Statistical features
  - Realized volatility
  - Hurst exponent
  - Lagged autocorrelation
  - Order flow imbalance proxy
- Microstructure features
  - Amihud illiquidity
  - Roll spread estimator
  - VWAP deviation
  - ATR
- Market structure
  - Trend strength
  - Volume confirmation
  - Breakout momentum

## Layer 3: Pattern and regime models

- `RegimeDetector`
  - Gaussian HMM
  - Kalman filtered trend
  - PELT breakpoints
- `PatternMatcher`
  - STUMPY matrix profile
  - DTW similarity search
- `TorchForecaster`
  - Lightweight convolutional return predictor

## Layer 4: Signal generation

- Ensemble scoring blends:
  - Regime conviction
  - Trend and momentum context
  - Volume confirmation
  - Motif / anomaly strength
  - Liquidity penalty
  - Optional deep forecast contribution

Outputs:

- Action label
- Confidence
- Entry zone
- Stop-loss
- Trailing stop
- Exit target
- Reasoning notes

## Layer 5: Backtesting

- `VectorBTBacktester`
  - Converts the strategy shell into historical entries / exits
  - Reports total return, Sharpe proxy, max drawdown, win rate, and trade count
