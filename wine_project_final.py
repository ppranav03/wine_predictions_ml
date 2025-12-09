# %% [markdown]
# # Wine Price Prediction Project
# 
# ## Description
# 
# 
# ## Table of Contents

# %% [markdown]
# ## 1) Library Imports and Data Downloading

# %%
# %% Imports
import os
import shutil as sh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kagglehub
import seaborn as sns
from tabulate import tabulate

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV

# Set global vars for cell output formatting.
LINE_BREAK = '=' * 50
LOAD_BREAK = '*' * 50


# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# %%
# %% Download datasets using kagglehub
path = kagglehub.dataset_download("zynicide/wine-reviews")
print(f"{LINE_BREAK}\n1) Path to dataset folder: {path}")

WINE_CSV_TITLE = "winemag-data_first150k.csv"
WINE_CSV_TITLE_2 = "winemag-data-130k-v2.csv"
first_path = os.path.join(path, WINE_CSV_TITLE)
second_path = os.path.join(path, WINE_CSV_TITLE_2)

print(LINE_BREAK)
print(f"2) Moving the CSVs to this directory\n{LOAD_BREAK}")

try:
    sh.move(first_path, ".")
    print(f"File {first_path} has been moved to this directory")
except FileNotFoundError:
    print("File was not found")
except Exception as e:
    print(e)

try:
    sh.move(second_path, ".")
    print(f"File {second_path} has been moved to this directory")
except FileNotFoundError:
    print("File was not found")
except Exception as e:
    print(e)

print(LINE_BREAK)

# %% Reading the dataset
print(f"3) Reading CSV {WINE_CSV_TITLE} and {WINE_CSV_TITLE_2}\n")
df = pd.read_csv(WINE_CSV_TITLE)
df2 = pd.read_csv(WINE_CSV_TITLE_2)

df = pd.concat([df, df2], ignore_index=True)
print("Raw shape:", df.shape)
print("Raw head:")
df.head()


# %% [markdown]
# ## 2) Basic Cleaning & Filtering

# %%
df.info()
print(LINE_BREAK + "\n")
# Combine region_1 and region_2 into a single 'region' column.
# If region_2 is missing, fall back to region_1.
df["region"] = df["region_2"].fillna(df["region_1"])

print(df.isna().sum().sort_values(ascending=False))
print(LINE_BREAK)
df = df.drop(columns=["region_2", "region_1", "taster_name", "taster_twitter_handle", df.columns[0], "title"])

print("Drop unecessary columns and rows with missing data.")
print(LINE_BREAK)
df = df.dropna(subset=["price", "region", "designation"])
null_columns = df.isna().sum().sort_values(ascending=False)
print(null_columns)

print(LINE_BREAK)

# For other important categoricals, fill missing with "Unknown"
fill_cols = ["country", "province", "variety", "region", "designation", "winery", "description"]
for col in fill_cols:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")


# %% Feature construction

# Create log_price as the regression target (to handle skew)
df["log_price"] = np.log1p(df["price"])  # log(1 + price)

print("Log transformation applied to price:")
print(LOAD_BREAK)
print(f"Original price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
print(f"Log price range: {df['log_price'].min():.3f} - {df['log_price'].max():.3f}")

# Keep only the columns we care about
# (you can add/remove here if you want to experiment)
needed_cols = [
    "log_price",
    "price",
    "points",
    "country",
    "province",
    "variety",
    "region",
    "description",
    "winery",
]
df = df[needed_cols]

print(LINE_BREAK)
print(f"\nFinal cleaned dataset size: {len(df)} rows")
print(f"Columns kept: {list(df.columns)}")

print("\nCleaned data preview:")
df.head()


# %% [markdown]
# ## 3) Exploratory Data Analysis (EDA)

# %%

# Price distribution
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# Original price distribution
axes[0].hist(df['price'], bins=50, edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Price ($)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('Distribution of Wine Prices', fontsize=14, fontweight='bold')
axes[0].axvline(df['price'].median(), color='red', linestyle='--', label=f'Median: ${df["price"].median():.2f}')
axes[0].legend()

# Log-transformed price distribution
axes[1].hist(df['log_price'], bins=50, edgecolor='black', alpha=0.7, color='green')
axes[1].set_xlabel('Log(Price)', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Distribution of Log-Transformed Prices', fontsize=14, fontweight='bold')
axes[1].axvline(df['log_price'].median(), color='red', linestyle='--', label=f'Median: {df["log_price"].median():.2f}')
axes[1].legend()

plt.tight_layout()
plt.show()

print(f"Price Statistics:")
print(f"  Mean: ${df['price'].mean():.2f}")
print(f"  Median: ${df['price'].median():.2f}")
print(f"  Std Dev: ${df['price'].std():.2f}")
print(f"  Skewness: {df['price'].skew():.2f}")


# %%
# %% Preprocessing: numeric, categorical, and text
numeric_features = ["points"]
categorical_features = ["country", "province", "variety", "region", "winery"]
text_feature = "description"

for col in categorical_features:
    freq = df[col].value_counts(normalize=True)
    df[col] = df[col].map(freq).astype(float)

print("After frequency encoding the categorical columns\n", tabulate(df.head()))


numeric_transformer = "passthrough"  # points is already numeric

categorical_transformer = "passthrough"
df[categorical_features] = df[categorical_features].fillna(0.0)
# categorical_transformer = OneHotEncoder(handle_unknown="ignore")

text_transformer = TfidfVectorizer(
    max_features=2000,
    stop_words="english"
)

# ColumnTransformer will:
# - pass 'points' through
# - one-hot encode country/province/variety/region
# - apply TF-IDF to description
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
        ("text", text_transformer, text_feature),
    ]
)

# %% Train / test split
# Features (X) and target (y). We will predict log_price.
feature_cols = numeric_features + categorical_features + [text_feature]
X = df[feature_cols]
y = df["log_price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTrain shape:", X_train.shape, "Test shape:", X_test.shape, "\n")

# %%
# %% Define models we want to try
models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.0005, max_iter=5000),
    "RandomForest": RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=10,
        n_jobs=-1,
        random_state=42
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),
}

ridge_params = {
    "model__alpha": [0.1, 1.0, 10.0, 50.0]
}

lasso_params = {
    "model__alpha": [0.0001, 0.001, 0.01],
    "model__max_iter": [3000, 5000]
}

rf_params = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [12, 15, 18],
    "model__min_samples_split": [5, 10, 20],
    "model__max_features": ["sqrt"]
}

gbr_params = {
    "model__n_estimators": [100, 200],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [2, 3]
}

param_grids = {
    "Ridge": ridge_params,
    "Lasso": lasso_params,
    "RandomForest": rf_params,
    "GradientBoosting": gbr_params
}

results = []

# %% Train and evaluate each model
for name, model in models.items():
    print(f"\n=== Training {name} ===")

    pipe = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    best_params = {}
    if name in param_grids:
        print(f"Running GridSearchCV for {name}...")
        grid = GridSearchCV(
            pipe,
            param_grids[name],
            cv=3,
            scoring="neg_mean_squared_error",
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        best_pipe = grid.best_estimator_
        best_params = grid.best_params_
        print(f"Best params: {best_params}")
    else:
        best_pipe = pipe.fit(X_train, y_train)

    # Predictions in log space
    y_train_pred_log = best_pipe.predict(X_train)
    y_test_pred_log = best_pipe.predict(X_test)

    # Convert back to original price scale
    y_train_price = np.expm1(y_train)
    y_test_price = np.expm1(y_test)

    y_train_pred_price = np.expm1(y_train_pred_log)
    y_test_pred_price = np.expm1(y_test_pred_log)

    train_rmse = root_mean_squared_error(y_train_price, y_train_pred_price)
    test_rmse = root_mean_squared_error(y_test_price, y_test_pred_price)
    train_mae = mean_absolute_error(y_train_price, y_train_pred_price)
    test_mae = mean_absolute_error(y_test_price, y_test_pred_price)
    train_r2 = r2_score(y_train_price, y_train_pred_price)
    test_r2 = r2_score(y_test_price, y_test_pred_price)
    
    print(f"Train RMSE: ${train_rmse:.2f} | Test RMSE: ${test_rmse:.2f}")
    print(f"Train MAE:  ${train_mae:.2f} | Test MAE:  ${test_mae:.2f}")
    print(f"Train R²:   {train_r2:.4f} | Test R²:   {test_r2:.4f}")
    
    results.append({
        "model": name,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "best_params": best_params
    })


# %%

# %% Wrap results into a DataFrame for easy viewing
results_df = pd.DataFrame(results).sort_values(by=["test_rmse", "train_rmse", "test_mae", "train_mae", "test_r2", "train_r2"])

print("\n=== Model Comparison (sorted by RMSE) ===")
print(tabulate(results_df))
print(LINE_BREAK)
best_line = results_df.iloc[0]
best_model_name = best_line["model"]
best_params = best_line["best_params"]


best_params = {k.replace("model__", ""): v for k, v in best_params.items()}
print(f"\n The best parameters for {best_model_name} are {best_params}")

# %%
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

x = np.arange(len(results_df))
width = 0.35

axes[0, 0].bar(x - width/2, results_df["train_rmse"], width=width, label = "Train", alpha=0.8)
axes[0, 0].bar(x + width/2, results_df["test_rmse"], width=width, label = "Test", alpha=0.8)
axes[0, 0].set_xlabel('Model', fontsize=12)
axes[0, 0].set_ylabel('RMSE ($)', fontsize=12)
axes[0, 0].set_title('RMSE Comparison', fontsize=14, fontweight='bold')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(results_df['model'], rotation=45, ha='right')
axes[0, 0].legend()
axes[0, 0].grid(axis='y', alpha=0.3)

axes[0, 1].bar(x - width/2, results_df["train_mae"], width=width, label="Train", alpha=0.8)
axes[0, 1].bar(x + width/2, results_df["test_mae"], width=width, label="Test", alpha=0.8)
axes[0, 1].set_xlabel('Model', fontsize=12)
axes[0, 1].set_ylabel('MAE ($)', fontsize=12)
axes[0, 1].set_title('MAE Comparison', fontsize=14, fontweight='bold')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(results_df['model'], rotation=45, ha='right')
axes[0, 1].legend()
axes[0, 1].grid(axis='y', alpha=0.3)

axes[1, 0].bar(x - width/2, results_df['train_r2'], width, label='Train', alpha=0.8)
axes[1, 0].bar(x + width/2, results_df['test_r2'], width, label='Test', alpha=0.8)
axes[1, 0].set_xlabel('Model', fontsize=12)
axes[1, 0].set_ylabel('R² Score', fontsize=12)
axes[1, 0].set_title('R² Score Comparison', fontsize=14, fontweight='bold')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(results_df['model'], rotation=45, ha='right')
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)

colors = ['green' if i == 0 else 'steelblue' for i in range(len(results_df))]
axes[1, 1].bar(results_df['model'], results_df['test_rmse'], color=colors, alpha=0.8)
axes[1, 1].set_xlabel('Model', fontsize=12)
axes[1, 1].set_ylabel('Test RMSE ($)', fontsize=12)
axes[1, 1].set_title('Test RMSE - Best Model Highlighted', fontsize=14, fontweight='bold')
axes[1, 1].set_xticklabels(results_df['model'], rotation=45, ha='right')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()


# %%

# === Plotting Section ===

match best_model_name:
    case "GradientBoosting":
        plot_model = GradientBoostingRegressor(**best_params)
    case "Ridge":
        plot_model = Ridge(**best_params)
    case "Lasso":
        plot_model = Lasso(**best_params)
    case "RandomForest":
        plot_model = RandomForestRegressor(**best_params)
    case "LinearRegression":
        plot_model = LinearRegression()
# Ensure best model is Linear Regression
best_pipe = Pipeline(steps=[("preprocessor", preprocessor),
                            ("model", plot_model)])
best_pipe.fit(X_train, y_train)

# Predictions
y_pred_log = best_pipe.predict(X_test)
y_pred_price = np.expm1(y_pred_log)
y_test_price = np.expm1(y_test)


# 1. Predicted vs Actual (Price)
plt.figure(figsize=(7, 7))
plt.scatter(y_test_price, y_pred_price, alpha=0.25, edgecolor='k', linewidth=0.2)
plt.plot([y_test_price.min(), y_test_price.max()], [y_test_price.min(), y_test_price.max()], color='red', linestyle='--', linewidth=2, label='Ideal')
plt.xlabel("Actual Price ($)", fontsize=13)
plt.ylabel("Predicted Price ($)", fontsize=13)
plt.title(f"{best_model_name} – Predicted vs Actual Prices", fontsize=15, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 2. Predicted vs Actual (log1p(price))
plt.figure(figsize=(7, 7))
plt.scatter(y_test, y_pred_log, alpha=0.25, edgecolor='k', linewidth=0.2)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", linewidth=2, label='Ideal')
plt.xlabel("Actual log1p(price)", fontsize=13)
plt.ylabel("Predicted log1p(price)", fontsize=13)
plt.title(f"{best_model_name} – Predicted vs Actual (log scale)", fontsize=15, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 3. Residuals vs Actual Price
residuals = y_test_price - y_pred_price
plt.figure(figsize=(7, 5))
plt.scatter(y_test_price, residuals, alpha=0.25, edgecolor='k', linewidth=0.2)
plt.axhline(0, color="red", linestyle="--", linewidth=2)
plt.xlabel("Actual Price ($)", fontsize=13)
plt.ylabel("Residual (Actual - Predicted)", fontsize=13)
plt.title(f"{best_model_name} – Residuals vs Actual Price", fontsize=15, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 4. Residual Histogram
plt.figure(figsize=(7, 5))
plt.hist(residuals, bins=75, alpha=0.7, color='steelblue', edgecolor='black')
plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Residual')
plt.xlabel("Residual (Actual - Predicted)", fontsize=13)
plt.ylabel("Frequency", fontsize=13)
plt.title(f"{best_model_name} – Residual Distribution", fontsize=15, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 5. Residuals vs Predicted Price
plt.figure(figsize=(7, 5))
plt.scatter(y_pred_price, residuals, alpha=0.25, edgecolor='k', linewidth=0.2)
plt.axhline(0, color='red', linestyle='--', linewidth=2)
plt.xlabel("Predicted Price ($)", fontsize=13)
plt.ylabel("Residual (Actual - Predicted)", fontsize=13)
plt.title(f"{best_model_name} – Residuals vs Predicted Price", fontsize=15, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%
print(LINE_BREAK)
print("Wine Price Prediction Project – Summary")
print(LINE_BREAK)

print("1. Data Overview:")
print(f"  - Total samples: {len(df)}")
print(f"  - Features used: {feature_cols}")
print(f"  - Target variable: log_price (log1p(price))")
print(f"  - Train samples: {len(X_train)}, Test samples: {len(X_test)}")
print(LINE_BREAK)

print("2. Models Evaluated:")
print(tabulate(results_df[["model", "train_rmse", "test_rmse", "train_mae", "test_mae", "train_r2", "test_r2"]], headers="keys", tablefmt="github"))
print(LINE_BREAK)

print("3. Best Model:")
print(f"  - Name: {best_model_name}")
print(f"  - Test RMSE: ${results_df.loc[results_df['model'] == best_model_name, 'test_rmse'].values[0]:.2f}")
print(f"  - Test MAE:  ${results_df.loc[results_df['model'] == best_model_name, 'test_mae'].values[0]:.2f}")
print(f"  - Test R²:   {results_df.loc[results_df['model'] == best_model_name, 'test_r2'].values[0]:.4f}")
print(f"  - Best Parameters: {best_params}")
print(LINE_BREAK)

print("4. Plots Generated:")
print("  - Price and log(price) distributions")
print("  - Model comparison bar charts (RMSE, MAE, R²)")
print("  - Predicted vs Actual prices (scatter)")
print("  - Residual analysis (scatter and histogram)")
print(LINE_BREAK)
print("Project complete. See above for details and visualizations.")


