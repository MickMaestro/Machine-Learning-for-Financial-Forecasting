# Machine Learning Financial Forecasting System

This code repository is a comprehensive automated trading system that integrates Python-based machine learning with MetaTrader 5 for enhanced financial market signal prediction and real-time trading execution.

## System Overview

This dissertation-level project presents a complete machine learning pipeline for financial trading that combines advanced technical analysis with artificial intelligence to improve trading signal accuracy. The system bridges the gap between Python-based machine learning research and practical trading implementation in MetaTrader 5.

### Core Architecture

The system employs a three-tier architecture:

1. **Data Generation Layer**: MetaTrader 5 scripts extract and preprocess historical trading data
2. **Machine Learning Layer**: Python-based training pipeline with multiple algorithms and ensemble methods  
3. **Trading Execution Layer**: MetaTrader 5 Expert Advisors with embedded ML models for live trading

### Key Innovation

The primary innovation lies in the seamless integration of complex Python ML models into real-time MQL5 trading systems through linear approximation and feature importance extraction, enabling sophisticated AI-driven trading decisions within MetaTrader 5's execution environment.

The deployed header is an *approximation*: a linear regression is fitted from the raw values of the top model features to the trained model's predicted probabilities, and the fit's fidelity (R² against the true model output) is embedded in every generated header so the quality of the simplification is always visible.

## Technical Specifications

### Programming Languages and Frameworks
- **Python 3.8+**: Core ML pipeline, data processing, and model training
- **MQL5**: MetaTrader 5 Expert Advisors and data exporters
- **Libraries**: scikit-learn, XGBoost, LightGBM, TensorFlow, pandas, numpy

### Machine Learning Algorithms
- **Random Forest**: Ensemble decision trees with balanced class weighting
- **XGBoost**: Gradient boosting with early stopping and regularisation  
- **LightGBM**: Fast gradient boosting optimised for large datasets
- **Support Vector Machine**: RBF kernel with probability estimates
- **LSTM Neural Networks**: Sequential pattern recognition for time series
- **Ensemble Methods**: Soft voting combinations of base algorithms

### Data Specifications
- **Input Format**: CSV files from the Data_Exporter2.0 system. MetaTrader 5 writes these as **UTF-16 with tab separators** — always load them through `RobustCSVLoader` (`src/utils/csv_loader.py`), which auto-detects encoding; a plain `pd.read_csv` will fail.
- **Feature Count**: 20 base model features + core signal columns (`Timestamp`, `Signal_Type`, `Entry_Price`, `Profit_Loss`) + derived engineered features created in Python
- **Data Sources**: Historical price data, technical indicators, market conditions
- **Labels**: Binary classification — `Signal_Quality = 1` if `Profit_Loss > 0`. Note that `Profit_Loss` is recorded in **raw price points of the instrument** (e.g. index points for GER40/NAS100), not forex pips.

## Project Structure

```
ml_financial_forecasting/
├── main.py                          # Primary application entry point
├── requirements.txt                 # Python dependency specifications
├── config/
│   ├── default_config.json         # System configuration parameters
│   └── __init__.py
├── src/                            # Core Python source code
│   ├── gui/
│   │   ├── main_interface.py       # Tkinter-based GUI implementation
│   │   └── __init__.py
│   ├── data_processing/
│   │   ├── preprocessor.py         # Data cleaning and validation
│   │   ├── feature_engineering.py  # Feature creation and selection
│   │   └── __init__.py
│   ├── trainers/                   # ML algorithm implementations
│   │   ├── base_trainer.py         # Abstract base class for all trainers
│   │   ├── random_forest_trainer.py
│   │   ├── xgboost_trainer.py
│   │   ├── lightgbm_trainer.py
│   │   ├── svm_trainer.py
│   │   ├── lstm_trainer.py
│   │   ├── ensemble_trainer.py
│   │   └── __init__.py
│   └── utils/
│       ├── config_manager.py       # Configuration management
│       ├── data_manager.py         # Data loading and management
│       ├── csv_loader.py           # CSV file handling utilities
│       ├── logging_utils.py        # Application logging framework
│       ├── mqh_generator.py        # MQL5 header file generation
│       ├── visualization.py        # Chart and plot generation
│       └── __init__.py
├── MQL5/                           # MetaTrader 5 source files
│   ├── Data_Exporter2.0.mq5       # Historical data extraction script
│   ├── ML_FF_AI.mq5               # Main Expert Advisor with ML filtering
│   ├── ML_FF_AI_CSV.mq5           # EA variant with CSV logging
│   ├── ML_FF_Basev2.mq5           # Baseline EA without ML
│   ├── Gann_EA.mq5                # Gann-only trading EA
│   ├── Gann_high-low_activator_2.mq5  # Core Gann indicator
│   └── Moving Average.mq5          # EMA trend filter indicator
├── MQH/                           # Generated ML model header files
│   ├── Ensemble_[SYMBOL]_[TIMEFRAME]_Model.mqh
│   ├── LightGBM_[SYMBOL]_[TIMEFRAME]_Model.mqh
│   ├── RandomForest_[SYMBOL]_[TIMEFRAME]_Model.mqh
│   ├── SVM_[SYMBOL]_[TIMEFRAME]_Model.mqh
│   └── XGBoost_[SYMBOL]_[TIMEFRAME]_Model.mqh
├── header_files/                   # Organised MQH files by symbol/timeframe
│   ├── GER40/
│   ├── NAS100/
│   ├── SP500/
│   └── US30/
├── data/
│   ├── raw/                       # Raw data storage
│   ├── signal/                    # Processed signal data by symbol
│   │   ├── GER40test/
│   │   ├── NAS100_TEST/
│   │   ├── USA30test/
│   │   └── USA500test/
│   └── exports/                   # Model prediction exports
├── models/                        # Trained model persistence
│   ├── [SYMBOL]/
│   │   ├── [ALGORITHM]_[TIMEFRAME]_[TIMESTAMP].pkl
│   │   └── [ALGORITHM]_[TIMEFRAME]_[TIMESTAMP]_features.txt
│   └── data_exporter_2_features_template.txt
├── plots/                         # Generated visualisations
│   ├── [SYMBOL]/
│   │   └── [ALGORITHM]/
│   │       └── [TIMEFRAME]/
└── logs/
    └── app.log                    # Application execution logs
```

## Data Processing Pipeline

### Data Export and Collection

The system begins with MetaTrader 5-based data collection using the Data_Exporter2.0 script. This MQL5 program simulates the complete GannWithEma trading strategy on historical data, generating training datasets.

**Key Features of Data_Exporter2.0:**
- Simulates Gann High-Low Activator 2 signal generation
- Applies dual EMA trend filtering (200-period median and low EMAs)
- Calculates the advanced technical analysis features per signal
- Tracks complete trade lifecycle from signal generation to outcome
- Exports structured CSV data with proper headers and timestamps

**Data Collection Process:**
1. Historical price data is loaded from MetaTrader 5 terminal
2. Gann indicators and EMA filters are calculated in real-time
3. Trading signals are identified based on strategy rules
4. For each signal, the feature set is computed and recorded
5. Trade simulation determines profit/loss outcomes
6. Results are exported to CSV files for ML training

> **Repository note:** the `MQL5/Data_Exporter2.0.mq5` file currently in this
> repository writes the *legacy 15-column format* (`EMA_Low`, `Gann_High`,
> `Volume`, …), not the full feature format found in `data/signal/`. The
> exporter version that produced the feature-format training CSVs is not
> checked in. Before generating new training data, either recover that
> exporter or extend the in-repo script — and verify the output columns
> against `data/signal/GER40test/GER40test_H1_signal_data.csv`.

### Feature Engineering Specifications

The system trains on **20 base technical analysis features** (categories 1–5
below), with the market condition columns (category 6) carried alongside as
core signal data — `Profit_Loss` is the label source and is never used as a
model input:

**1. Gann Zone Features (6 features):**
- `Gann_Zone_Width_Pips`: Width of active Gann trading zone
- `Gann_Zone_Slope`: Slope indicating zone direction
- `Price_Distance_From_Zone_Center`: Price position relative to zone centre
- `Price_Distance_From_Zone_Boundary`: Distance to nearest zone boundary
- `Active_Gann_Line_Slope`: Slope of current active Gann line
- `Bars_Since_Last_Gann_Color_Change`: Timing of last zone transition

**2. EMA Features (5 features):**
- `EMA_Separation_Pips`: Distance between 200-period median and low EMAs
- `Median_EMA_Slope`: Trend direction of median EMA
- `Low_EMA_Slope`: Trend direction of low EMA
- `Price_Distance_From_Relevant_EMA`: Price position relative to trend EMA
- `EMA_Convergence_Divergence`: EMA relationship indicator

**3. Temporal Features (3 features):**
- `Trading_Session`: Market session classification — **1=Asian/off-hours, 2=London (08–12h), 4=London/NY overlap (13–15h), 3=NY (16–21h)**. This exact encoding is used in the training data and mirrored in the EAs' `calculateMLFeatures()`; changing it on either side silently invalidates every trained model.
- `Day_Of_Week`: Day classification (Monday=1 through Friday=5)
- `Hour_Of_Day`: Hour of signal generation (0-23)

**4. Candle Analysis Features (4 features):**
- `Entry_Candle_Size_Pips`: Size of entry candle in pips
- `Entry_Candle_Body_Ratio`: Ratio of candle body to total range (0-1)
- `Previous_Candle_Size_Pips`: Previous candle size for context
- `Bars_Since_Relevant_EMA_Cross`: Timing of last EMA crossover

**5. Risk Management Features (2 features):**
- `Initial_SL_Distance_Pips`: Distance to initial stop loss
- `Recent_Gann_Signal_Frequency`: Signal density in recent periods

**6. Market Condition Columns (core signal data, not model inputs):**
- `Spread`: Bid-ask spread at signal time (not present in the historical training CSVs)
- `Signal_Type`: Direction indicator (Buy=1, Sell=-1)
- `Entry_Price`: Actual entry price level
- `Profit_Loss`: Trade outcome in raw price points — the source of the `Signal_Quality` label, never a feature

### Derived Feature Engineering

The FeatureEngineer class creates additional derived features from the base feature set:

**Trading Session Binary Indicators:**
- `Is_Asian_Session`, `Is_London_Session`, `Is_NY_Session`, `Is_Overlap_Session`

**Temporal Pattern Indicators:**
- `Is_Monday`, `Is_Friday`, `Is_Midweek`
- `Is_Market_Open_Hours`, `Is_Late_Night`

**Signal Quality Indicators:**
- `Is_Buy_Signal`, `Is_Sell_Signal`
- `Relative_Position_In_Zone`, `EMA_Slopes_Alignment`

**Risk Classification Features:**
- `Is_Tight_Stop`, `Is_Wide_Stop`
- `High_Signal_Activity`, `Low_Signal_Activity`

**Market Condition Classifications:**
- `High_Spread`, `Low_Spread`

> **Caveat — quantile-based derived features:** the risk/activity/spread
> classifications above are computed from quantiles of the *whole loaded
> dataset*. They are useful for offline research, but (a) they let summary
> statistics of the test period leak into training features, and (b) they
> cannot be reproduced bar-by-bar inside MetaTrader 5. They are therefore
> excluded from anything deployed to the EA: the MQH generator only selects
> from features the EA can genuinely calculate at signal time.

**Interaction Features:**
- `Gann_EMA_Alignment`: Alignment between Gann zones and EMA trends
- `Favorable_Time_Conditions`: Optimal timing combinations
- `Momentum_Alignment`: Directional agreement across indicators
- `Fresh_Signal`: Recent signal generation indicator

## Machine Learning Implementation

### Training Architecture

The system employs an object-oriented training architecture with a common base class (`BaseTrainer`) and algorithm-specific implementations. Each trainer inherits core functionality while implementing algorithm-specific optimisations.

**Common Training Pipeline:**
1. Data validation and preprocessing
2. Signal labelling: `Signal_Quality = 1` if `Profit_Loss > 0` (price points)
3. Feature engineering and selection
4. **Chronological** train/test splitting (80/20 ratio) — every trainer splits by time, never by shuffling, so no future information reaches the training set
5. Hyperparameter optimisation with `TimeSeriesSplit` cross-validation (validation folds always come after their training folds)
6. Model training and evaluation
7. Confidence threshold optimisation (F1-optimal threshold over a 0.10–0.90 grid)
8. Model persistence including everything the MQH generator needs (`X_test`, `feature_importance`, `optimal_threshold`, metrics)

### Algorithm Implementations and Core Mechanics

**Random Forest Algorithm:**

*How it Works:*
Random Forest is an ensemble method that combines multiple decision trees to create a robust predictor. Each tree is trained on a random subset of the data (bootstrap sampling) and uses a random subset of features at each split point. The algorithm works through:

1. **Bootstrap Sampling**: Create multiple datasets by randomly sampling with replacement from the original training data
2. **Feature Randomness**: At each node split, randomly select a subset of features to consider for the best split
3. **Tree Construction**: Build decision trees using the sampled data and feature subsets, with each tree voting on the final prediction
4. **Aggregation**: Combine predictions through majority voting (classification) or averaging (regression)

*Financial Trading Application:*
Random Forest excels in financial applications because it can capture non-linear relationships between technical indicators without overfitting. The feature importance ranking helps identify which technical factors (Gann zones, EMA slopes, market timing) are most predictive of profitable signals.

*Implementation Details:*
- GridSearchCV explores estimators (50–200), depth (5–15) and split/leaf minimums, scored by F1 with `TimeSeriesSplit(5)` so validation data always postdates training data
- Balanced class weighting addresses the imbalance between profitable and unprofitable signals
- Feature importance extraction enables interpretable model decisions

**XGBoost Algorithm:**

*How it Works:*
XGBoost (Extreme Gradient Boosting) is an advanced gradient boosting framework that builds models sequentially, where each new model corrects errors made by previous models. The algorithm operates through:

1. **Sequential Learning**: Start with an initial prediction (usually the mean) and iteratively add new weak learners
2. **Gradient Calculation**: Calculate gradients and hessians of the loss function for each training example
3. **Tree Construction**: Build decision trees that best reduce the current residual errors using second-order optimisation
4. **Regularisation**: Apply L1/L2 penalties to prevent overfitting and improve generalisation
5. **Boosting**: Add new trees with scaled contributions (learning rate) to the ensemble

*Mathematical Foundation:*
The objective function combines prediction accuracy with regularisation:
`Obj = Σ L(yi, ŷi) + Σ Ω(fk)`
where L is the loss function, Ω is the regularisation term, and fk represents individual trees.

*Financial Trading Application:*
XGBoost's ability to handle complex feature interactions makes it particularly suitable for financial data where relationships between indicators are non-linear and time-dependent. The gradient-based optimisation effectively learns from trading signal patterns.

*Implementation Details:*
- Learning rate of 0.1 provides stable convergence
- Subsample ratio 0.8 introduces randomness to prevent overfitting
- L1/L2 regularisation balances model complexity and performance
- Early stopping prevents overtraining on historical patterns

**LightGBM Algorithm:**

*How it Works:*
LightGBM is a gradient boosting framework optimised for speed and memory efficiency. It differs from traditional gradient boosting through:

1. **Leaf-wise Growth**: Grows trees by adding leaves that reduce loss most, rather than level-wise growth
2. **Histogram-based Learning**: Uses histograms to find optimal splits, reducing memory usage and computational cost
3. **Feature Bundling**: Combines sparse features to reduce the number of features without losing information
4. **Gradient-based One-Side Sampling**: Focuses on samples with larger gradients for more efficient learning

*Optimisation Innovations:*
- Uses histogram-based algorithms for faster split finding
- Implements exclusive feature bundling for sparse feature spaces
- Employs gradient-based one-side sampling to reduce data size whilst maintaining accuracy

*Financial Trading Application:*
LightGBM's speed advantages make it ideal for rapid model iteration and backtesting. The leaf-wise growth strategy effectively captures the hierarchical nature of trading decisions (market conditions → signal generation → profitability).

*Implementation Details:*
- Leaf-wise tree growth for optimal accuracy with fewer leaves
- Built-in categorical feature handling for market sessions and time periods
- Memory-efficient processing enables training on large historical datasets
- Automatic feature importance calculation guides feature selection

**Support Vector Machine (SVM) Algorithm:**

*How it Works:*
SVM finds the optimal decision boundary (hyperplane) that separates classes with maximum margin. For non-linearly separable data, it uses kernel functions to transform features into higher-dimensional spaces where linear separation becomes possible:

1. **Margin Maximisation**: Find the hyperplane that maximises the distance to the nearest training examples (support vectors)
2. **Kernel Trick**: Use kernel functions (RBF, polynomial, linear) to handle non-linear relationships without explicitly computing high-dimensional transformations
3. **Soft Margin**: Allow some misclassifications through slack variables when perfect separation is impossible
4. **Quadratic Optimisation**: Solve the constrained optimisation problem to find optimal hyperplane parameters

*Mathematical Foundation:*
The SVM optimisation problem minimises:
`½||w||² + C Σ ξi`
subject to classification constraints, where w is the weight vector, C is the regularisation parameter, and ξi are slack variables.

*Financial Trading Application:*
SVM's ability to handle high-dimensional feature spaces makes it effective for financial data with many technical indicators. The RBF kernel captures non-linear relationships between features like Gann zones and EMA configurations.

*Implementation Details:*
- RBF kernel handles complex non-linear feature relationships
- Balanced class weighting addresses signal imbalance in financial data
- Probability estimates enable confidence-based trading decisions
- Grid search optimisation finds optimal C and gamma parameters

**LSTM Neural Network Algorithm:**

*How it Works:*
Long Short-Term Memory networks are recurrent neural networks designed to learn long-term temporal dependencies. LSTMs solve the vanishing gradient problem through specialised memory cells:

1. **Cell State**: Maintains long-term memory through a conveyor belt mechanism
2. **Forget Gate**: Decides what information to discard from cell state using sigmoid activation
3. **Input Gate**: Determines what new information to store in cell state
4. **Output Gate**: Controls what parts of cell state to output to hidden state
5. **Sequential Processing**: Processes time series data step by step, maintaining memory of previous states

*Architecture Components:*
- **Input Gate**: `it = σ(Wxi·xt + Whi·ht-1 + bi)`
- **Forget Gate**: `ft = σ(Wxf·xt + Whf·ht-1 + bf)`
- **Output Gate**: `ot = σ(Wxo·xt + Who·ht-1 + bo)`
- **Cell State Update**: `Ct = ft * Ct-1 + it * tanh(Wxc·xt + Whc·ht-1 + bc)`

*Financial Trading Application:*
LSTMs excel at capturing temporal patterns in market data, learning from sequences of price movements and indicator values. They can identify momentum patterns and market regime changes that influence signal profitability.

*Implementation Details:*
- 50 hidden units provide sufficient capacity for pattern recognition
- 10-period sequence length captures short to medium-term market memory
- 0.2 dropout prevents overfitting on historical sequences
- Adam optimiser with 0.001 learning rate ensures stable convergence

**Ensemble Algorithm:**

*How it Works:*
Ensemble methods combine predictions from multiple base algorithms to create a more robust and accurate predictor. The system implements two ensemble approaches:

1. **Voting Ensemble**: Combines predictions through weighted voting
   - Hard Voting: Each classifier votes for a class, majority wins
   - Soft Voting: Averages predicted probabilities for more nuanced decisions
   
2. **Stacking Ensemble**: Uses a meta-learner to optimally combine base model predictions
   - Base models make predictions on cross-validated data
   - Meta-learner (Random Forest) learns optimal combination strategy
   - Reduces overfitting through careful validation methodology

*Theoretical Foundation:*
Ensemble methods leverage the bias-variance trade-off:
- **Bias Reduction**: Combining different algorithms reduces systematic errors
- **Variance Reduction**: Averaging predictions reduces sensitivity to training data variations
- **Diversity Benefit**: Different algorithms capture different aspects of the underlying patterns

*Financial Trading Application:*
Ensemble methods are particularly valuable in financial markets where no single approach consistently outperforms. By combining tree-based methods (Random Forest, XGBoost) with SVM and neural networks, the ensemble captures diverse trading signal patterns.

*Implementation Details:*
- Soft voting uses probability estimates for more refined decisions
- Cross-validation ensures unbiased meta-learner training
- Weighted averaging based on individual model validation performance
- Dynamic weight adjustment based on recent model performance

### Model Evaluation Framework

The system employs comprehensive evaluation metrics:

**Classification Metrics:**
- Accuracy: Overall prediction correctness
- Precision: True positive rate for profitable signals
- Recall: Proportion of profitable signals correctly identified
- F1-Score: Harmonic mean of precision and recall
- ROC-AUC: Area under receiver operating characteristic curve

**Financial Performance Metrics:**
- Sharpe Ratio: Risk-adjusted return measure
- Maximum Drawdown: Largest peak-to-trough decline
- Win Rate: Percentage of profitable trades
- Average Win/Loss Ratio: Risk-reward relationship

**Threshold Optimisation:**
The system performs confidence threshold analysis to optimise the trade-off between signal frequency and accuracy. Multiple threshold values are evaluated to find the optimal balance for each trading strategy.

## MetaTrader 5 Integration

### MQH Header Generation

The MQHGenerator class converts trained Python models into MQL5-compatible header files through linear approximation. The trained models consume ~40 scaled features, while the EA computes ~20 raw features live, so the generator bridges both gaps explicitly:

**Conversion Process:**
1. Load trained model, scaler, feature column list and test data from the joblib file
2. Compute the model's predicted probabilities on the held-out test set, **applying the saved StandardScaler first** (the models were trained on scaled data — predicting on raw values would produce meaningless output). LightGBM `Booster`, Keras and all sklearn-API models are handled.
3. Rank features by importance (or |correlation| with the predictions for models without native importances, e.g. SVM) and keep the top 10 — restricted to features the EA can genuinely calculate at signal time, matched **by name** to the EA's 24-slot `features[]` array
4. Fit a linear regression from the **raw** values of those features to the model probabilities — the resulting coefficients operate in raw-feature space, so the EA applies them directly without a scaler
5. Record the fit's R² against the true model probabilities and embed it in the header, so the fidelity of the approximation is always visible
6. Emit the MQL5 header with EA-space feature indices, coefficients, bias, optimal threshold and real test metrics

If a legacy model file lacks test data, the generator falls back to heuristic coefficients — and stamps a prominent `WARNING: HEURISTIC FALLBACK` block into the header so it can never be mistaken for a model-derived filter.

**Generated Header Structure:**
```cpp
// Linear approximation fidelity (R^2 vs model probabilities): 0.5213

// Model metadata and performance metrics
#define MODEL_ALGORITHM "Random Forest"
#define MODEL_SYMBOL "GER40"
#define MODEL_TIMEFRAME "H1"
#define MODEL_THRESHOLD 0.300000
#define MODEL_ACCURACY 0.9945

// Indices into the EA's 24-element features[] array
static int FEATURE_INDICES[10] = {...};
// Coefficients (raw feature space - applied directly to EA-calculated values)
static double FEATURE_COEFFS[10] = {...};
static double MODEL_BIAS = -0.0094;

// Main evaluation functions
bool evaluateMLConfidence(double &features[])   // confidence >= MODEL_THRESHOLD
double getConfidenceScore(double &features[])   // clamp(bias + sum(coef*feature), 0, 1)
```

Note that the confidence score is a clamped linear approximation of a probability. The R² line at the top of each header quantifies how faithful that approximation is — values well below 1.0 mean the EA is trading a simplified shadow of the model, which is the main structural limitation of this deployment method.

### Expert Advisor Integration

The ML_FF_AI Expert Advisor implements the complete trading system:

**Core Components:**
- Gann High-Low Activator 2 for signal generation
- Dual EMA trend filtering system
- Real-time feature calculation matching training data
- ML confidence evaluation using included header
- Risk management with Gann-based stop losses

**Trading Logic Flow:**
1. Monitor market for Gann signal generation
2. Validate signal with EMA trend filter
3. Calculate the 24-slot real-time feature array (`calculateMLFeatures()`)
4. Evaluate ML confidence using embedded model
5. Execute trade only if confidence exceeds threshold
6. Manage position with adaptive stop losses

**Model Selection Process:**
1. Copy desired .mqh file to MQL5/Include directory
2. Uncomment appropriate #include statement in EA
3. Recompile EA to embed model logic
4. Configure confidence threshold parameter
5. Deploy on live or backtesting environment

## Installation and Setup

### System Requirements

**Software Prerequisites:**
- Python 3.8 or higher
- MetaTrader 5 terminal (for trading integration)
- Minimum 8GB RAM for model training
- 2GB available storage for data and models

**Python Dependencies:**
```
pandas>=1.5.0          # Data manipulation and analysis
numpy>=1.21.0          # Numerical computing
scikit-learn>=1.1.0    # Machine learning algorithms
xgboost>=1.6.0         # Gradient boosting
lightgbm>=3.3.0        # Fast gradient boosting
tensorflow>=2.10.0     # Deep learning (optional)
matplotlib>=3.5.0      # Visualisation
joblib>=1.1.0          # Model persistence
chardet>=5.0.0         # Character encoding detection
```

### Installation Process

**Step 1: Environment Preparation**
```bash
# Clone or download the repository
git clone [repository_url]
cd ml_financial_forecasting

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

**Step 2: Configuration Setup**
```bash
# Verify configuration file
cat config/default_config.json

# Create necessary directories
mkdir -p data/signal
mkdir -p models
mkdir -p plots
mkdir -p logs
```

**Step 3: Data Preparation**
```bash
# Place CSV files from Data_Exporter2.0 in appropriate directories
cp your_signal_data.csv data/signal/[SYMBOL]/

# Verify data format (MT5 CSVs are UTF-16 - use the project loader, not plain pandas)
python -c "from src.utils.csv_loader import RobustCSVLoader; ok, df, info = RobustCSVLoader().load_csv_intelligent('data/signal/[SYMBOL]/[FILE].csv'); print(df.columns.tolist())"
```

### Usage Instructions

**GUI Application Launch:**
```bash
python main.py
```

**Command-Line Model Training:**
```python
from src.utils.csv_loader import RobustCSVLoader
from src.trainers.random_forest_trainer import RandomForestTrainer
from src.utils.mqh_generator import generate_mqh_from_model

# Load the MT5-exported CSV (UTF-16/tab handled automatically)
loader = RobustCSVLoader()
ok, df, info = loader.load_csv_intelligent(
    "data/signal/GER40test/GER40test_H1_signal_data.csv")

# Train: labelling, feature engineering, chronological split,
# grid search, threshold analysis and prediction export in one call
trainer = RandomForestTrainer()
trainer.df = df.copy()
trainer.run_complete_training_with_gui()

# Persist the model (includes X_test/importance/threshold for MQH generation)
trainer.save_model("models/GER40/rf_H1.pkl")

# Generate the MQL5 header
generate_mqh_from_model("models/GER40/rf_H1.pkl", "GER40", "H1",
                        output_dir="header_files/GER40/H1")
```

### Configuration Management

The system uses JSON-based configuration for flexibility:

**Key Configuration Sections:**
- **Application Metadata**: Version, author, description
- **Data Processing**: File paths, validation rules, feature specifications
- **Training Parameters**: Algorithm hyperparameters, cross-validation settings
- **Model Management**: Persistence options, versioning, performance tracking
- **Visualisation**: Plot settings, colour schemes, output formats
- **GUI Settings**: Window dimensions, themes, refresh intervals
- **Logging Configuration**: Log levels, file rotation, format specifications

**Example Configuration Modification:**
```json
{
  "training": {
    "hyperparameters": {
      "Random Forest": {
        "n_estimators": 300,
        "max_depth": 20,
        "min_samples_split": 2,
        "class_weight": "balanced"
      }
    }
  }
}
```

## Data Flow Architecture

### Complete Pipeline Overview

**1. Historical Data Generation (MetaTrader 5)**
```
Market Data → Data_Exporter2.0.mq5 → CSV Files → data/signal/
```

**2. Machine Learning Training (Python)**
```
CSV Data → Feature Engineering → Model Training → .pkl Files → models/
```

**3. Model Conversion (Python)**
```
.pkl Models → MQH Generator → .mqh Headers → MQH/
```

**4. Live Trading Deployment (MetaTrader 5)**
```
.mqh Headers → ML_FF_AI.mq5 → Live Trading → Results/
```

### Data Storage Organisation

**Signal Data Structure:**
- `data/signal/[SYMBOL]/[SYMBOL]_[TIMEFRAME]_signal_data.csv`
- Contains complete feature set and trade outcomes
- Organised by trading symbol and timeframe

**Model Storage Structure:**
- `models/[SYMBOL]/[ALGORITHM]_[TIMEFRAME]_[TIMESTAMP].pkl`
- Includes corresponding feature list files (.txt)
- Timestamped for version control

**Export Data Structure:**
- `data/exports/[algorithm]_predictions_[timestamp].csv`
- Real-time prediction outputs for analysis
- Used for model validation and performance tracking

**Generated Headers Structure:**
- `MQH/[ALGORITHM]_[SYMBOL]_[TIMEFRAME]_Model.mqh`
- Ready for MetaTrader 5 integration
- Contains embedded model logic and metadata

### File Generation Patterns

**Model Training Output:**
Each training session generates:
- Trained model file (.pkl)
- Feature importance list (.txt)
- Performance visualisations (.png)
- Training log entries
- Optional MQH header file

**Prediction Export Output:**
Each prediction run generates:
- Timestamped CSV file with predictions
- Confidence scores for each signal
- Model metadata and parameters
- Performance metrics summary

**Visualisation Output:**
Charts are generated for:
- Feature importance rankings
- ROC curves and threshold analysis
- Performance metrics comparison
- Training progress monitoring

## Advanced Features and Capabilities

### Ensemble Learning Implementation

The system implements sophisticated ensemble methods:

**Voting Ensemble:**
- Combines predictions from multiple base algorithms
- Soft voting uses probability estimates for final decision
- Weights can be adjusted based on individual model performance
- Cross-validation ensures robust ensemble performance

**Stacking Ensemble:**
- Uses meta-learner to combine base model predictions
- Second-level Random Forest learns optimal combination strategy
- Reduces overfitting through cross-validated predictions
- Achieves superior performance compared to individual models

### Hyperparameter Optimisation

**Grid Search Implementation:**
- Systematic exploration of hyperparameter space
- Cross-validation for reliable performance estimation
- Time-based splitting respects temporal nature of financial data
- Automated selection of optimal parameters

**Algorithm-Specific Optimisation:**
- Random Forest: Tree depth, number of estimators, splitting criteria
- XGBoost: Learning rate, regularisation, tree parameters
- SVM: Kernel parameters, regularisation strength
- LSTM: Network architecture, dropout rates, sequence length

### Model Validation Framework

**Time Series Cross-Validation:**
- Respects temporal ordering of financial data
- Walk-forward analysis for realistic performance estimation
- Multiple validation windows for robust evaluation
- Prevents data leakage from future to past

**Out-of-Sample Testing:**
- Strict chronological splitting for realistic evaluation
- Hold-out test set never used during training
- Performance metrics calculated on unseen data
- Confidence intervals for statistical significance

### Feature Selection and Engineering

**Automated Feature Selection:**
- Importance-based feature ranking
- Recursive feature elimination
- Correlation analysis to remove redundant features
- Stability selection for robust feature sets

**Domain-Specific Engineering:**
- Financial market expertise embedded in feature design
- Technical analysis principles guide feature creation
- Market microstructure considerations
- Risk management integration

## Performance Analysis and Results

### Evaluation Methodology

The system employs rigorous performance evaluation:

**Statistical Metrics:**
- Classification accuracy with confidence intervals
- Precision and recall for imbalanced datasets
- F1-score for balanced performance assessment
- ROC-AUC for threshold-independent evaluation

**Financial Metrics:**
- Profit factor (gross profit / gross loss)
- Maximum drawdown analysis
- Sharpe ratio for risk-adjusted returns
- Win rate and average win/loss ratios

**Model Comparison:**
- Side-by-side algorithm performance
- Statistical significance testing
- Ensemble vs individual model analysis
- Threshold sensitivity analysis

### Visualisation Capabilities

**Performance Charts:**
- ROC curves with AUC values
- Precision-recall curves
- Confusion matrices with class distributions
- Learning curves showing training progress

**Feature Analysis:**
- Feature importance rankings
- Correlation heatmaps
- Distribution analysis
- Temporal stability assessment

**Trading Performance:**
- Equity curves and drawdown analysis
- Signal distribution over time
- Confidence score distributions
- Trade outcome analysis

### Model Interpretability

**Feature Importance Analysis:**
- Algorithm-specific importance metrics
- SHAP values for individual predictions
- Permutation importance for robustness
- Partial dependence plots for feature relationships

**Decision Boundary Analysis:**
- Threshold optimisation curves
- Confidence distribution analysis
- Signal quality segmentation
- Risk-return trade-off visualisation

## Risk Management Integration

### Position Sizing Framework

The system incorporates sophisticated risk management:

**Dynamic Position Sizing:**
- ML confidence-based position scaling
- Volatility-adjusted lot sizes
- Account balance percentage limits
- Maximum exposure constraints

**Stop Loss Management:**
- Gann-level based stop placement
- Trailing stops for trend following
- Confidence-based stop adjustment
- Risk-reward ratio optimisation

### Model Risk Controls

**Overfitting Prevention:**
- Cross-validation with temporal splits
- Regularisation in all algorithms
- Feature selection for dimensionality reduction
- Ensemble methods for robustness

**Performance Monitoring:**
- Real-time model degradation detection
- Confidence score distribution monitoring
- Prediction accuracy tracking
- Automated retraining triggers

## Future Development and Extensions

### Planned Enhancements

**Algorithm Expansion:**
- Deep reinforcement learning for adaptive strategies
- Transformer models for sequence modelling
- Gaussian process regression for uncertainty quantification
- Online learning for real-time adaptation

**Feature Engineering:**
- Alternative data integration (sentiment, news)
- Multi-timeframe feature synthesis
- Volatility clustering models
- Market regime detection

**Integration Improvements:**
- Real-time data streaming
- Cloud-based model serving
- Mobile application interface
- Portfolio-level risk management

### Research Opportunities

**Academic Extensions:**
- Explainable AI for financial decisions
- Causal inference in trading strategies
- Multi-objective optimisation
- Quantum machine learning applications

**Industry Applications:**
- High-frequency trading integration
- Multi-asset strategy development
- Risk factor modelling
- Regulatory compliance automation

## Known Limitations and Data Integrity Notes

Honest constraints to keep in mind when interpreting results or extending the system:

**Deployment fidelity.** The MQH header is a linear approximation of the trained model. Measured on this project's data, the approximation explains roughly R² 0.2–0.5 of the true model probabilities depending on algorithm (reported in each generated header). Conclusions about *model* performance from Python evaluation do not automatically transfer to *EA* performance — always validate the deployed header in the Strategy Tester.

**Class imbalance.** The test datasets are heavily imbalanced (e.g. GER40 H1: 31 profitable signals out of 1,808). Accuracy is therefore near-meaningless on this data; judge models by precision/recall/F1 at the chosen confidence threshold, and treat metrics computed from a handful of positive samples as fragile.

**Exporter version gap.** The in-repo `Data_Exporter2.0.mq5` writes the legacy 15-column CSV format; the exporter that produced the feature-format CSVs in `data/signal/` is not checked in (see the Data Export section).

**Encoding contracts.** Three conventions must stay in lockstep between the exporter, the Python pipeline and the EAs, or every deployed model silently breaks: the Trading_Session hour mapping (1/2/4/3 — see Temporal Features), the feature order of the EA's `features[]` array, and pip conversion (`price difference / (Point()*10)`).

**Stale generated headers.** Any `.mqh` file generated before the 2026-07 pipeline fixes (below) predates the corrected generator — identifiable by `MODEL_ACCURACY 0.0000` and round heuristic coefficients — and should be regenerated from retrained models before use.

## Changelog — 2026-07 Pipeline Fixes

A full review of the pipeline fixed the following defects (relevant when comparing against results produced before this date):

**Training correctness:**
- `create_signal_labels` previously multiplied `Profit_Loss` by 10,000 (a forex pip conversion) before applying tiered thresholds; on index data this collapsed to a sign check with misleading intermediate logic. Labelling is now an explicit sign check on price points.
- The LSTM trainer's feature list included `Profit_Loss` (the label source) and `Entry_Price` — direct target leakage. It now uses the same feature engineering path as every other trainer.
- The Random Forest trainer used a shuffled `train_test_split` and shuffled CV folds, leaking future data into training. It now splits chronologically and cross-validates with `TimeSeriesSplit`.
- The Ensemble trainer substituted *random* feature importances when averaging failed; this fallback was removed.

**Deployment correctness:**
- The MQH generator selected feature indices in the ~40-column training feature space but the EA interpreted them against its 24-slot array — coefficients were applied to the wrong features. Features are now mapped by name.
- The generator fed unscaled data to models trained on scaled data when building the linear approximation. Predictions are now computed with the saved scaler, and the linear fit is performed in raw-feature space so the EA needs no scaler.
- LightGBM `Booster` models crashed the generator (`predict_proba` missing); LightGBM/Ensemble model files lacked the data needed for generation, silently producing heuristic headers with `MODEL_ACCURACY 0.0000`. All trainers now persist the complete generation payload, and any fallback header carries an explicit warning block.
- The generated header now reports the linear approximation's R² against the true model probabilities.

**MQL5 correctness:**
- `ArrayInitialise` (not an MQL5 function) corrected to `ArrayInitialize` in `ML_FF_AI.mq5` and `ML_FF_AI_CSV.mq5`.
- The EAs computed `Trading_Session` as 0–3 while all training data uses 1–4 (with different hour boundaries); worse, the Python preprocessor treats session 0 as invalid. The EAs now reproduce the training encoding exactly.

## Conclusion

This machine learning financial forecasting system represents a comprehensive solution for automated trading strategy development. By combining sophisticated Python-based machine learning with practical MetaTrader 5 integration, it bridges the gap between academic research and real-world trading applications.

The system's modular architecture, extensive feature engineering, and robust validation framework provide a solid foundation for financial machine learning research and practical trading system development. The automated conversion of complex ML models to MQL5 headers enables sophisticated AI-driven trading decisions within standard retail trading platforms.

## Technical Support and Documentation

### Logging and Debugging

The system provides comprehensive logging:

**Application Logs:**
- Located in `logs/app.log`
- Configurable log levels (DEBUG, INFO, WARNING, ERROR)
- Automatic log rotation to prevent disk space issues
- Structured logging with timestamps and module identification

**Debug Information:**
- Model training progress tracking
- Feature engineering status updates
- Data validation results
- Performance metric calculations

### Error Handling

**Graceful Degradation:**
- Missing data handling with imputation
- Model loading failure recovery
- GUI error display with user guidance
- Configuration validation with helpful messages

**Data Quality Checks:**
- CSV format validation
- Feature completeness verification
- Temporal consistency checking
- Outlier detection and handling

### Configuration Validation

The system validates configuration files on startup:
- Required parameter presence checking
- Data type validation
- Range validation for numerical parameters
- Path existence verification

### Performance Optimisation

**Memory Management:**
- Efficient data loading with chunking
- Model memory usage optimisation
- Garbage collection for large datasets
- Resource cleanup after training

**Computational Efficiency:**
- Parallel processing where applicable
- Optimised feature engineering
- Cached intermediate results
- GPU acceleration for LSTM training (when available)

## Contact and Attribution

**Author:** Ibukunoluwa Michael Adebanjo  
**Institution:** University of Essex  
**Project Type:** Dissertation Research Project  
**Academic Year:** 2025

This system represents original research in machine learning applications for financial markets, combining theoretical foundations with practical implementation for automated trading strategy development.

---
