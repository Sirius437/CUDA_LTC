#pragma once

#include <vector>
#include <string>
#include <map>
#include <limits>
#include <algorithm>
#include <cmath>

// Data structure for price bars with technical indicators
struct Bar {
    std::string date;
    // Keep price data as double for precision in calculations
    double open = 0.0;
    double high = 0.0;
    double low = 0.0;
    double close = 0.0;
    double volume = 0.0;
    
    // Technical indicators in FP32 format (changed from FP16)
    float vwap = 0.0f;
    float ema5 = 0.0f;
    float ema9 = 0.0f;
    float ema11 = 0.0f;
    float ema20 = 0.0f;
    float ema32 = 0.0f;
    float bb_upper = 0.0f;
    float bb_middle = 0.0f;
    float bb_lower = 0.0f;
    float rsi = 0.0f;
    float macd = 0.0f;
    float macd_signal = 0.0f;
    float macd_hist = 0.0f;
    float stoch_k = 0.0f;
    float stoch_d = 0.0f;
    
    // Additional indicators in FP32 format (changed from FP16)
    float atr = 0.0f;           // Average True Range
    float atr_percent = 0.0f;   // ATR as percentage of price
    float obv = 0.0f;           // On-Balance Volume
    float sar = 0.0f;           // Parabolic SAR
    float adx = 0.0f;           // Average Directional Index
    float cci = 0.0f;           // Commodity Channel Index
    float mfi = 0.0f;           // Money Flow Index
    float cmo = 0.0f;           // Chande Momentum Oscillator
    float willr = 0.0f;         // Williams %R
    float ultosc = 0.0f;        // Ultimate Oscillator
    
    // NEW: Price & Trend indicators (12 features)
    float ema3 = 0.0f;          // EMA(3) - very short term
    float ema8 = 0.0f;          // EMA(8) - short term
    float ema13 = 0.0f;         // EMA(13) - Fibonacci period
    float dema = 0.0f;          // Double Exponential Moving Average
    float tema = 0.0f;          // Triple Exponential Moving Average
    float kama = 0.0f;          // Kaufman Adaptive Moving Average
    float linear_reg = 0.0f;    // Linear Regression
    float linearreg_slope = 0.0f; // Linear Regression Slope
    float tsf = 0.0f;           // Time Series Forecast
    
    // NEW: Volatility & Range indicators (5 features)
    float natr = 0.0f;          // Normalized Average True Range
    float trange = 0.0f;        // True Range
    float stddev = 0.0f;        // Standard Deviation
    float variance = 0.0f;      // Variance
    
    // NEW: Momentum & Oscillators (2 new features)
    float roc = 0.0f;           // Rate of Change
    float stochrsi = 0.0f;      // Stochastic RSI
    
    // NEW: Volume & Flow indicators (3 features)
    float adosc = 0.0f;         // Chaikin A/D Oscillator
    float chaikin_money_flow = 0.0f; // Chaikin Money Flow
    float obv_delta = 0.0f;     // OBV change from previous period
    
    // NEW: Hilbert Transform / Cycle indicators (3 features)
    float ht_dcperiod = 0.0f;   // Hilbert Transform - Dominant Cycle Period
    float ht_dcphase = 0.0f;    // Hilbert Transform - Dominant Cycle Phase
    float ht_sine = 0.0f;       // Hilbert Transform - Sine Wave
    
    // NEW: Volatility Ratios (1 feature)
    float atr_ratio = 0.0f;     // ATR relative to its moving average
    
    // NEW: Candlestick Patterns (3 features)
    float cdlhammer = 0.0f;     // Hammer pattern strength
    float cdlengulfing = 0.0f;  // Engulfing pattern strength
    float cdldoji = 0.0f;       // Doji pattern strength
    
    // NEW: Permutation Entropy (1 feature)
    float perm_entropy = 0.0f;  // Permutation entropy for complexity
    
    // NEW: Portfolio Features (4 features - populated externally)
    float portfolio_position = 0.0f;  // Current position size
    float portfolio_value = 0.0f;     // Total portfolio value
    float portfolio_cash = 0.0f;      // Available cash
    float portfolio_shares = 0.0f;    // Number of shares held
    
    // NEW: Engineered Features (6 features)
    float signal_line = 0.0f;         // Custom signal line (crossover logic)
    float delta_kama = 0.0f;          // KAMA momentum
    float delta_linear_reg = 0.0f;    // Linear regression momentum
    float delta_obv = 0.0f;           // OBV momentum (same as obv_delta)
    float price_percent_change = 0.0f; // Price percentage change
    float volume_change_ratio = 0.0f;  // Volume change ratio
    
    // NEW: Cross-asset/Market Features (4 features)
    float correl_asset_x = 0.0f;      // Correlation with market index
    float beta_asset_x = 0.0f;        // Beta relative to market
    float market_index_rsi = 0.0f;    // Market index RSI
    float market_index_atr = 0.0f;    // Market index ATR
    
    // Extra TA-Lib indicators (8 features)
    float trix = 0.0f;                // TRIX oscillator
    float plus_di = 0.0f;             // Plus Directional Indicator
    float minus_di = 0.0f;            // Minus Directional Indicator
    float plus_dm = 0.0f;             // Plus Directional Movement
    float minus_dm = 0.0f;            // Minus Directional Movement
    float aroon = 0.0f;               // AROON indicator
    float aroonosc = 0.0f;            // AROON Oscillator
    float bop = 0.0f;                 // Balance of Power
    
    // Trading signals in FP32 format (changed from FP16)
    float long_score = 0.0f;
    float short_score = 0.0f;
    int trading_signal = -1;    // 1 = buy, 0 = sell, -1 = hold/neutral
    
    // Additional flags
    bool near_support = false;
    bool near_resistance = false;
    bool increasing_volume = false;
};

// Normalized data structure for machine learning/neural network input
struct NormalizedData {
    std::string date;                  // Original date for reference
    float actual_price = 0.0f;         // Actual price (reference price used for normalization)
    
    // All values normalized to appropriate ranges for ML input (changed from FP16 to FP32)
    float norm_close = 0.0f;        // close / current_price
    float norm_volume = 0.0f;       // volume / 1e6
    float norm_ema5 = 0.0f;         // ema5 / current_price
    float norm_ema9 = 0.0f;         // ema9 / current_price
    float norm_ema11 = 0.0f;        // ema11 / current_price
    float norm_ema20 = 0.0f;        // ema20 / current_price
    float norm_ema32 = 0.0f;        // ema32 / current_price
    float norm_bb_upper = 0.0f;     // bb_upper / current_price
    float norm_bb_middle = 0.0f;    // bb_middle / current_price
    float norm_bb_lower = 0.0f;     // bb_lower / current_price
    float norm_macd_hist = 0.0f;    // macd_hist / current_price
    float norm_rsi = 0.0f;          // rsi / 100.0 (0-1)
    float norm_macd = 0.0f;         // macd / current_price
    float norm_macd_signal = 0.0f;  // macd_signal / current_price
    float norm_stoch_k = 0.0f;      // stoch_k / 100.0 (0-1)
    float norm_stoch_d = 0.0f;      // stoch_d / 100.0 (0-1)
    float norm_vwap = 0.0f;         // vwap / current_price
    float norm_atr = 0.0f;          // atr / current_price
    float norm_atr_percent = 0.0f;  // atr_percent / 10.0 (0-1)
    float norm_obv = 0.0f;          // obv / 1e6
    float norm_sar = 0.0f;          // sar / current_price
    float norm_perm_entropy = 0.0f; // Permutation Entropy (already 0-1)
    float norm_cmo = 0.0f;          // (cmo + 100) / 200.0 (0-1)
    float norm_willr = 0.0f;        // willr / -100.0 (0-1)
    float norm_signal = 0.0f;       // (signal + 1) / 2.0 (0-1, -1 to +1)
    float norm_near_support = 0.0f; // Binary: 0 or 1
    float norm_near_resistance = 0.0f; // Binary: 0 or 1
    float norm_inc_volume = 0.0f;   // Binary: 0 or 1
    float norm_cci = 0.0f;          // (cci + 200) / 400.0 (0-1)
    float norm_adx = 0.0f;          // adx / 100.0 (0-1)
    float norm_mfi = 0.0f;          // mfi / 100.0 (0-1)
    float norm_ultosc = 0.0f;       // ultosc / 100.0 (0-1)
    
    // Extra TA-Lib indicators (8 features)
    float norm_trix = 0.0f;         // (trix + 0.01) / 0.02 (normalized around 0)
    float norm_plus_di = 0.0f;      // plus_di / 100.0 (0-1)
    float norm_minus_di = 0.0f;     // minus_di / 100.0 (0-1)
    float norm_plus_dm = 0.0f;      // plus_dm / current_price
    float norm_minus_dm = 0.0f;     // minus_dm / current_price
    float norm_aroon = 0.0f;        // aroon / 100.0 (0-1)
    float norm_aroonosc = 0.0f;     // (aroonosc + 100) / 200.0 (0-1)
    float norm_bop = 0.0f;          // (bop + 1.0) / 2.0 (0-1)
    
    // NEW: Price & Trend normalized features (9 features)
    float norm_ema3 = 0.0f;         // ema3 / current_price
    float norm_ema8 = 0.0f;         // ema8 / current_price
    float norm_ema13 = 0.0f;        // ema13 / current_price
    float norm_dema = 0.0f;         // dema / current_price
    float norm_tema = 0.0f;         // tema / current_price
    float norm_kama = 0.0f;         // kama / current_price
    float norm_linear_reg = 0.0f;   // linear_reg / current_price
    float norm_linearreg_slope = 0.0f; // linearreg_slope / current_price * 100
    float norm_tsf = 0.0f;          // tsf / current_price
    
    // NEW: Volatility & Range normalized features (5 features)
    float norm_natr = 0.0f;         // natr / 100.0 (0-1, already normalized)
    float norm_trange = 0.0f;       // trange / current_price
    float norm_stddev = 0.0f;       // stddev / current_price
    float norm_variance = 0.0f;     // variance / (current_price^2)
    float norm_atr_ratio = 0.0f;    // atr_ratio (already ratio, 0-5 typical)
    
    // NEW: Momentum & Oscillators normalized features (3 features)
    float norm_roc = 0.0f;          // (roc + 50) / 100.0 (0-1, -50% to +50%)
    float norm_stochrsi = 0.0f;     // stochrsi / 100.0 (0-1)
    float norm_macd_diff = 0.0f;    // (macd - macd_signal) / current_price
    
    // NEW: Volume & Flow normalized features (4 features)
    float norm_adosc = 0.0f;        // adosc / 1e6 (similar to volume)
    float norm_chaikin_money_flow = 0.0f; // (cmf + 1) / 2.0 (0-1, -1 to +1)
    float norm_obv_delta = 0.0f;    // obv_delta / 1e6
    float norm_volume_percent_change = 0.0f; // (vol_change + 2) / 4.0 (0-1, -200% to +200%)
    
    // NEW: Hilbert Transform normalized features (3 features)
    float norm_ht_dcperiod = 0.0f;  // ht_dcperiod / 50.0 (0-1, typical 10-50 period)
    float norm_ht_dcphase = 0.0f;   // (ht_dcphase + 180) / 360.0 (0-1, -180 to +180)
    float norm_ht_sine = 0.0f;      // (ht_sine + 1) / 2.0 (0-1, -1 to +1)
    
    // NEW: Candlestick Patterns normalized features (3 features)
    float norm_cdlhammer = 0.0f;    // (cdlhammer + 100) / 200.0 (0-1, -100 to +100)
    float norm_cdlengulfing = 0.0f; // (cdlengulfing + 100) / 200.0 (0-1, -100 to +100)
    float norm_cdldoji = 0.0f;      // (cdldoji + 100) / 200.0 (0-1, -100 to +100)
    
    // NEW: Portfolio normalized features (4 features)
    float portfolio_position = 0.0f;    // Raw position (can be negative for short)
    float portfolio_value = 0.0f;       // portfolio_value / 1e6 (millions)
    float portfolio_cash = 0.0f;        // portfolio_cash / 1e6 (millions)
    float portfolio_shares = 0.0f;      // portfolio_shares / 1000.0 (thousands)
    
    // NEW: Engineered Features normalized (6 features)
    float signal_line = 0.0f;           // (signal_line + 1) / 2.0 (0-1, -1 to +1)
    float delta_kama = 0.0f;            // delta_kama / current_price * 100
    float delta_linear_reg = 0.0f;      // delta_linear_reg / current_price * 100
    float delta_obv = 0.0f;             // delta_obv / 1e6 (same as obv_delta)
    float price_percent_change = 0.0f;  // (price_change + 0.1) / 0.2 (0-1, -10% to +10%)
    float volume_change_ratio = 0.0f;   // volume_change_ratio / 5.0 (0-1, 0 to 5x)
    
    // NEW: Cross-asset normalized features (4 features)
    float correl_asset_x = 0.0f;        // (correlation + 1) / 2.0 (0-1, -1 to +1)
    float beta_asset_x = 0.0f;          // beta / 3.0 (0-1, 0 to 3.0 typical)
    float market_index_rsi = 0.0f;      // market_rsi / 100.0 (0-1)
    float market_index_atr = 0.0f;      // market_atr / market_price
    
    // Constructor to normalize data from a Bar
    NormalizedData(const Bar& bar, double current_price = 0.0) {
        // If current_price is not provided, use the bar's close price
        if (current_price <= 0.0) {
            current_price = bar.close;
        }
        
        date = bar.date;
        actual_price = static_cast<float>(current_price);
        
        // Price normalization
        norm_close = bar.close / current_price;
        norm_volume = bar.volume / 1e6;
        
        // EMAs normalized by price
        norm_ema5 = bar.ema5 / current_price;
        norm_ema9 = bar.ema9 / current_price;
        norm_ema11 = bar.ema11 / current_price;
        norm_ema20 = bar.ema20 / current_price;
        norm_ema32 = bar.ema32 / current_price;
        
        // Bollinger Bands normalized by price
        norm_bb_upper = bar.bb_upper / current_price;
        norm_bb_middle = bar.bb_middle / current_price;
        norm_bb_lower = bar.bb_lower / current_price;
        
        // MACD components normalized by price
        norm_macd = bar.macd / current_price;
        norm_macd_signal = bar.macd_signal / current_price;
        norm_macd_hist = bar.macd_hist / current_price;
        
        // Oscillators normalized to 0-1 range
        norm_rsi = bar.rsi / 100.0f;
        norm_stoch_k = bar.stoch_k / 100.0f;
        norm_stoch_d = bar.stoch_d / 100.0f;
        
        // Other indicators normalized
        norm_vwap = bar.vwap / current_price;
        norm_atr = bar.atr / current_price;
        norm_atr_percent = bar.atr_percent / 10.0f;
        norm_obv = bar.obv / 1e6;
        norm_sar = bar.sar / current_price;
        
        // Normalize CMO from -100/+100 to 0-1
        norm_cmo = (bar.cmo + 100.0f) / 200.0f;
        
        // Normalize Williams %R from -100/0 to 0-1
        norm_willr = bar.willr / -100.0f;
        
        // Normalize CCI from -200/+200 to 0-1
        norm_cci = (bar.cci + 200.0f) / 400.0f;
        
        // Normalize ADX, MFI, and Ultimate Oscillator from 0-100 to 0-1
        norm_adx = bar.adx / 100.0f;
        norm_mfi = bar.mfi / 100.0f;
        norm_ultosc = bar.ultosc / 100.0f;
        
        // Normalize trading signal from -1,0,1 to 0-1
        norm_signal = (bar.trading_signal + 1.0f) / 2.0f;
        
        // Binary indicators (already 0 or 1)
        norm_near_support = bar.near_support ? 1.0f : 0.0f;
        norm_near_resistance = bar.near_resistance ? 1.0f : 0.0f;
        norm_inc_volume = bar.increasing_volume ? 1.0f : 0.0f;
        
        // Permutation Entropy (already in 0-1 range)
        norm_perm_entropy = bar.perm_entropy;
        
        // NEW: Price & Trend normalized features
        norm_ema3 = bar.ema3 / current_price;
        norm_ema8 = bar.ema8 / current_price;
        norm_ema13 = bar.ema13 / current_price;
        norm_dema = bar.dema / current_price;
        norm_tema = bar.tema / current_price;
        norm_kama = bar.kama / current_price;
        norm_linear_reg = bar.linear_reg / current_price;
        norm_linearreg_slope = bar.linearreg_slope / current_price * 100;
        norm_tsf = bar.tsf / current_price;
        
        // NEW: Volatility & Range normalized features
        norm_natr = bar.natr / 100.0f;
        norm_trange = bar.trange / current_price;
        norm_stddev = bar.stddev / current_price;
        norm_variance = bar.variance / (current_price * current_price);
        norm_atr_ratio = bar.atr_ratio;
        
        // NEW: Momentum & Oscillators normalized features
        norm_roc = (bar.roc + 50.0f) / 100.0f;
        norm_stochrsi = bar.stochrsi / 100.0f;
        norm_macd_diff = (bar.macd - bar.macd_signal) / current_price;
        
        // NEW: Volume & Flow normalized features
        norm_adosc = bar.adosc / 1e6;
        norm_chaikin_money_flow = (bar.chaikin_money_flow + 1.0f) / 2.0f;
        norm_obv_delta = bar.obv_delta / 1e6;
        norm_volume_percent_change = (bar.volume_change_ratio + 2.0f) / 4.0f;
        
        // NEW: Hilbert Transform normalized features
        norm_ht_dcperiod = bar.ht_dcperiod / 50.0f;
        norm_ht_dcphase = (bar.ht_dcphase + 180.0f) / 360.0f;
        norm_ht_sine = (bar.ht_sine + 1.0f) / 2.0f;
        
        // NEW: Candlestick Patterns normalized features
        norm_cdlhammer = (bar.cdlhammer + 100.0f) / 200.0f;
        norm_cdlengulfing = (bar.cdlengulfing + 100.0f) / 200.0f;
        norm_cdldoji = (bar.cdldoji + 100.0f) / 200.0f;
        
        // NEW: Portfolio normalized features
        portfolio_position = bar.portfolio_position;
        portfolio_value = bar.portfolio_value / 1e6;
        portfolio_cash = bar.portfolio_cash / 1e6;
        portfolio_shares = bar.portfolio_shares / 1000.0f;
        
        // NEW: Engineered Features normalized
        signal_line = (bar.signal_line + 1.0f) / 2.0f;
        delta_kama = bar.delta_kama / current_price * 100;
        delta_linear_reg = bar.delta_linear_reg / current_price * 100;
        delta_obv = bar.delta_obv / 1e6;
        price_percent_change = (bar.price_percent_change + 0.1f) / 0.2f;
        volume_change_ratio = bar.volume_change_ratio / 5.0f;
        
        // NEW: Cross-asset normalized features
        correl_asset_x = (bar.correl_asset_x + 1.0f) / 2.0f;
        beta_asset_x = bar.beta_asset_x / 3.0f;
        market_index_rsi = bar.market_index_rsi / 100.0f;
        market_index_atr = bar.market_index_atr / current_price;
        
        // Extra TA-Lib indicators
        norm_trix = (bar.trix + 0.01f) / 0.02f;
        norm_plus_di = bar.plus_di / 100.0f;
        norm_minus_di = bar.minus_di / 100.0f;
        norm_plus_dm = bar.plus_dm / current_price;
        norm_minus_dm = bar.minus_dm / current_price;
        norm_aroon = bar.aroon / 100.0f;
        norm_aroonosc = (bar.aroonosc + 100.0f) / 200.0f;
        norm_bop = (bar.bop + 1.0f) / 2.0f;
    }
    
    // Default constructor
    NormalizedData() = default;
};