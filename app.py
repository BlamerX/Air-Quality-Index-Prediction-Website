import os
import re
import json
import time
import threading
import warnings
import numpy as np
import pandas as pd
import requests
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from bs4 import BeautifulSoup, Comment
from concurrent.futures import ThreadPoolExecutor

# Data processing
from sklearn.impute import KNNImputer
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    RobustScaler,
    PowerTransformer,
    LabelEncoder,
)
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Machine Learning Models (CPU-compatible)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsRegressor

# Advanced Models (CPU-compatible)
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# Utilities
import joblib
import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Settings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", None)
plt.style.use("seaborn-v0_8-dark-palette")
sns.set_palette("husl")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Global variables for training status
training_status = {
    "current_phase": "idle",
    "current_step": "Ready to start",
    "progress": 0,
    "logs": [],
    "training_complete": False,
    "best_model": None,
    "model_metrics": {},
    "all_models_data": {},
    "feature_importances": [],
}

# Lock for thread-safe operations
training_lock = threading.Lock()

# Create directories if they don't exist
os.makedirs("saved_models", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

# CPU-only configuration
print("Running on CPU")


# Data extraction functions
def extract_city_data(city_url):
    """Extract air quality data for a single city"""
    try:
        city_response = requests.get(city_url, timeout=10)
        city_response.raise_for_status()
        city_soup = BeautifulSoup(city_response.content, "html.parser")

        # Extract city and state names
        city_name = city_soup.find("h2").text.strip()
        state_info = city_soup.find("p").text.strip().split(", ")
        state_name = state_info[1] if len(state_info) > 1 else "Unknown"

        # Extract AQI value from HTML comment
        aqi_comment = city_soup.find(
            text=lambda text: isinstance(text, Comment) and "indexValue" in text
        )
        aqi_value = (
            float(re.search(r"\d+", aqi_comment).group()) if aqi_comment else np.NaN
        )

        # Extract pollutant information
        pollutants = city_soup.find_all("div", class_="pollutant-item")
        pollutant_data = {}
        for pollutant in pollutants:
            name = pollutant.find("div", class_="name").text.strip()
            value = pollutant.find("div", class_="value").text.strip()
            pollutant_data[name] = float(value) if value else np.NaN

        # Extract weather information
        temperature = city_soup.find("div", class_="temperature")
        temperature = (
            float(temperature.text.strip().replace("℃", "")) if temperature else np.NaN
        )
        humidity = city_soup.find("div", class_="humidity")
        humidity = float(humidity.text.strip().replace("%", "")) if humidity else np.NaN
        wind_speed = city_soup.find("div", class_="wind")
        wind_speed = (
            float(wind_speed.text.strip().replace("kph", "")) if wind_speed else np.NaN
        )

        aqi_type = city_soup.find("div", class_="level")
        aqi_type = aqi_type.text.strip() if aqi_type else "Unknown"
        aqi_type = aqi_type.replace("Moderately polluted", "Moderate")

        return {
            "State": state_name,
            "City": city_name,
            "PM2.5": pollutant_data.get("PM2.5"),
            "PM10": pollutant_data.get("PM10"),
            "O3": pollutant_data.get("O3"),
            "SO2": pollutant_data.get("SO2"),
            "CO": pollutant_data.get("CO"),
            "NO2": pollutant_data.get("NO2"),
            "Wind Speed": wind_speed,
            "Humidity": humidity,
            "Temperature": temperature,
            "AQI": aqi_value,
            "AQI Type": aqi_type,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
    except Exception as e:
        print(f"Error extracting data from {city_url}: {str(e)}")
        return None


def scrape_all_cities(base_url, max_workers=50):
    """Scrape air quality data for all cities"""
    global training_status
    with training_lock:
        training_status["current_step"] = "Starting data extraction..."
        training_status["logs"].append("Starting data extraction...")

    start_time = datetime.now()
    try:
        response = requests.get(base_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        site_items = soup.find_all("a", class_="site-item")

        data = []
        processed_cities = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for site_item in site_items:
                state_href = site_item["href"]
                try:
                    state_response = requests.get(state_href, timeout=10)
                    state_response.raise_for_status()
                    state_soup = BeautifulSoup(state_response.content, "html.parser")
                    city_hrefs = state_soup.find_all("a", class_="site-item")

                    for city_href in city_hrefs:
                        city_url = city_href["href"]
                        if "https://air-quality.com/place/india" in city_url:
                            future = executor.submit(extract_city_data, city_url)
                            futures.append(future)
                            processed_cities += 1
                            if processed_cities % 10 == 0:
                                with training_lock:
                                    progress = min(
                                        25, int((processed_cities / 100) * 25)
                                    )
                                    training_status["progress"] = progress
                                    training_status["current_step"] = (
                                        f"Processed {processed_cities} cities..."
                                    )
                                    training_status["logs"].append(
                                        f"Processed {processed_cities} cities..."
                                    )
                except Exception as e:
                    print(f"Error processing state {state_href}: {str(e)}")
                    continue

            # Collect results
            for future in futures:
                result = future.result()
                if result:
                    data.append(result)

        end_time = datetime.now()
        extraction_time = end_time - start_time

        with training_lock:
            training_status["progress"] = 25
            training_status["current_step"] = (
                f"Data extraction completed in {extraction_time}"
            )
            training_status["logs"].append(
                f"Data extraction completed in {extraction_time}"
            )
            training_status["logs"].append(f"Total cities processed: {len(data)}")

        return pd.DataFrame(data)
    except Exception as e:
        with training_lock:
            training_status["current_step"] = f"Error in scraping: {str(e)}"
            training_status["logs"].append(f"Error in scraping: {str(e)}")
        return pd.DataFrame()


# Data processing functions
def handle_missing_values(df):
    """Handle missing values using advanced imputation techniques"""
    with training_lock:
        training_status["current_step"] = "Handling missing values..."
        training_status["logs"].append("Handling missing values...")

    # Separate numerical and categorical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    # For numerical columns, use KNN imputation
    imputer = KNNImputer(n_neighbors=5, weights="distance")
    df[numerical_cols] = imputer.fit_transform(df[numerical_cols])

    # For categorical columns, use mode imputation
    for col in categorical_cols:
        if col != "Timestamp":  # Skip timestamp column
            mode_value = df[col].mode()[0]
            df[col] = df[col].fillna(mode_value)

    with training_lock:
        training_status["progress"] = 35
        training_status["logs"].append("Missing values handled successfully")

    return df


def create_features(df):
    """Create additional features for better model performance"""
    with training_lock:
        training_status["current_step"] = "Creating additional features..."
        training_status["logs"].append("Creating additional features...")

    # Create AQI categories based on ranges
    def get_aqi_category(aqi):
        if aqi <= 50:
            return "Good"
        elif aqi <= 100:
            return "Satisfactory"
        elif aqi <= 200:
            return "Moderate"
        elif aqi <= 300:
            return "Poor"
        elif aqi <= 400:
            return "Very Poor"
        else:
            return "Severe"

    df["AQI Category"] = df["AQI"].apply(get_aqi_category)

    # Create pollutant ratios
    df["PM2.5_PM10_Ratio"] = df["PM2.5"] / (df["PM10"] + 1e-6)
    df["SO2_NO2_Ratio"] = df["SO2"] / (df["NO2"] + 1e-6)

    # Create composite pollution index
    df["Composite_Pollution_Index"] = (
        df["PM2.5"] * 0.3
        + df["PM10"] * 0.2
        + df["SO2"] * 0.15
        + df["NO2"] * 0.15
        + df["CO"] * 0.1
        + df["O3"] * 0.1
    )

    # Create weather impact score
    df["Weather_Impact"] = (
        df["Temperature"] * 0.3 + df["Humidity"] * 0.4 + df["Wind Speed"] * 0.3
    )

    # Encode categorical variables
    le_state = LabelEncoder()
    le_city = LabelEncoder()
    le_aqi_type = LabelEncoder()
    le_aqi_category = LabelEncoder()

    df["State_Encoded"] = le_state.fit_transform(df["State"])
    df["City_Encoded"] = le_city.fit_transform(df["City"])
    df["AQI_Type_Encoded"] = le_aqi_type.fit_transform(df["AQI Type"])
    df["AQI_Category_Encoded"] = le_aqi_category.fit_transform(df["AQI Category"])

    with training_lock:
        training_status["progress"] = 50
        training_status["logs"].append("Feature engineering completed")

    return df, le_state, le_city, le_aqi_type, le_aqi_category


def prepare_features(df):
    """Prepare features for modeling"""
    # Select numerical features
    numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove target variable and non-predictive features
    features_to_remove = ["AQI", "Timestamp"]
    numerical_features = [f for f in numerical_features if f not in features_to_remove]

    X = df[numerical_features]
    y = df["AQI"]

    return X, y, numerical_features


# Model training functions


def evaluate_models(models, X_train, X_test, y_train, y_test):
    """Evaluate multiple models and return results"""
    results = {}
    total_models = len(models)
    completed_models = 0

    for name, model in models.items():
        with training_lock:
            training_status["current_step"] = f"Training {name}..."
            training_status["logs"].append(f"Training {name}...")

        # Train model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Cross-validation
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring="neg_mean_squared_error"
        )
        cv_mse = -cv_scores.mean()

        results[name] = {
            "MSE": mse,
            "MAE": mae,
            "R2": r2,
            "CV_MSE": cv_mse,
            "model": model,
            "model_type": type(model).__name__,
        }

        completed_models += 1
        progress = 50 + int((completed_models / total_models) * 40)

        with training_lock:
            training_status["progress"] = progress
            training_status["logs"].append(
                f"{name} - MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}, CV MSE: {cv_mse:.4f}"
            )

    return results


def get_feature_importance(model, feature_names):
    """Get feature importance from model"""
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        importance_df = pd.DataFrame(
            {"Feature": feature_names, "Importance": importance}
        ).sort_values("Importance", ascending=False)
        return importance_df
    else:
        return None


def save_model(model, model_name, scaler, feature_names, encoders):
    """Save model and related objects"""
    # Create directory if it doesn't exist
    os.makedirs("saved_models", exist_ok=True)

    # Save model
    joblib.dump(model, f"saved_models/{model_name}.pkl")

    # Save scaler
    joblib.dump(scaler, f"saved_models/{model_name}_scaler.pkl")

    # Save feature names
    joblib.dump(feature_names, f"saved_models/{model_name}_features.pkl")

    # Save encoders
    joblib.dump(encoders, f"saved_models/{model_name}_encoders.pkl")

    print(f"Model saved as saved_models/{model_name}")


def load_model(model_name):
    """Load model and related objects"""
    # Load model
    model = joblib.load(f"saved_models/{model_name}.pkl")

    # Load scaler
    scaler = joblib.load(f"saved_models/{model_name}_scaler.pkl")

    # Load feature names
    feature_names = joblib.load(f"saved_models/{model_name}_features.pkl")

    # Load encoders
    encoders = joblib.load(f"saved_models/{model_name}_encoders.pkl")

    return model, scaler, feature_names, encoders


def predict_aqi(input_data, model, scaler, feature_names, encoders):
    """
    Predict AQI for new data
    """
    # Create DataFrame from input
    input_df = pd.DataFrame([input_data])

    # For categorical features, use the most common value from training if 'Unknown' is passed
    if input_data["State"] == "Unknown":
        input_df["State"] = encoders["state"].classes_[
            0
        ]  # Use first state from training
    if input_data["City"] == "Unknown":
        input_df["City"] = encoders["city"].classes_[0]  # Use first city from training
    if input_data["AQI Type"] == "Unknown":
        input_df["AQI Type"] = encoders["aqi_type"].classes_[
            0
        ]  # Use first AQI type from training

    # Encode categorical variables
    try:
        input_df["State_Encoded"] = encoders["state"].transform([input_df["State"]])[0]
    except:
        input_df["State_Encoded"] = 0

    try:
        input_df["City_Encoded"] = encoders["city"].transform([input_df["City"]])[0]
    except:
        input_df["City_Encoded"] = 0

    try:
        input_df["AQI_Type_Encoded"] = encoders["aqi_type"].transform(
            [input_df["AQI Type"]]
        )[0]
    except:
        input_df["AQI_Type_Encoded"] = 0

    # Calculate AQI category
    aqi = input_data.get("AQI", 0)
    if aqi <= 50:
        category = "Good"
    elif aqi <= 100:
        category = "Satisfactory"
    elif aqi <= 200:
        category = "Moderate"
    elif aqi <= 300:
        category = "Poor"
    elif aqi <= 400:
        category = "Very Poor"
    else:
        category = "Severe"

    try:
        input_df["AQI_Category_Encoded"] = encoders["aqi_category"].transform(
            [category]
        )[0]
    except:
        input_df["AQI_Category_Encoded"] = 0

    # Create engineered features
    input_df["PM2.5_PM10_Ratio"] = input_df["PM2.5"] / (input_df["PM10"] + 1e-6)
    input_df["SO2_NO2_Ratio"] = input_df["SO2"] / (input_df["NO2"] + 1e-6)
    input_df["Composite_Pollution_Index"] = (
        input_df["PM2.5"] * 0.3
        + input_df["PM10"] * 0.2
        + input_df["SO2"] * 0.15
        + input_df["NO2"] * 0.15
        + input_df["CO"] * 0.1
        + input_df["O3"] * 0.1
    )
    input_df["Weather_Impact"] = (
        input_df["Temperature"] * 0.3
        + input_df["Humidity"] * 0.4
        + input_df["Wind Speed"] * 0.3
    )

    # Select features
    X = input_df[feature_names]

    # Scale features
    X_scaled = scaler.transform(X)

    # Make prediction
    prediction = model.predict(X_scaled)

    return prediction[0]


def get_aqi_type(aqi):
    """Get AQI type from AQI value"""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"


def get_aqi_color(aqi):
    """Get AQI color from AQI value"""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 200:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 300:
        return "Unhealthy"
    elif aqi <= 400:
        return "Very Unhealthy"
    else:
        return "Hazardous"


def get_aqi_advice(aqi):
    """Get health advice based on AQI value"""
    if aqi <= 50:
        return {
            "level": "Good",
            "color": "#00e400",
            "advice": "Air quality is satisfactory, and air pollution poses little or no risk.",
            "health_implications": "None",
            "cautionary_statement": "None",
        }
    elif aqi <= 100:
        return {
            "level": "Moderate",
            "color": "#ffff00",
            "advice": "Air quality is acceptable. However, there may be a risk for some people.",
            "health_implications": "Sensitive individuals may experience minor symptoms.",
            "cautionary_statement": "Sensitive people should consider reducing prolonged outdoor exertion.",
        }
    elif aqi <= 200:
        return {
            "level": "Unhealthy for Sensitive Groups",
            "color": "#ff7e00",
            "advice": "Members of sensitive groups may experience health effects.",
            "health_implications": "Increased likelihood of respiratory symptoms in sensitive individuals.",
            "cautionary_statement": "Sensitive groups should reduce outdoor activities.",
        }
    elif aqi <= 300:
        return {
            "level": "Unhealthy",
            "color": "#ff0000",
            "advice": "Everyone may begin to experience health effects.",
            "health_implications": "Increased aggravation of heart or lung disease.",
            "cautionary_statement": "Everyone should reduce prolonged outdoor exertion.",
        }
    elif aqi <= 400:
        return {
            "level": "Very Unhealthy",
            "color": "#8f3f97",
            "advice": "Health alert: everyone may experience serious health effects.",
            "health_implications": "Increased aggravation of heart or lung disease and premature mortality.",
            "cautionary_statement": "Avoid outdoor activities. Keep windows closed.",
        }
    else:
        return {
            "level": "Hazardous",
            "color": "#7e0023",
            "advice": "Health warnings of emergency conditions.",
            "health_implications": "Everyone is likely to be affected.",
            "cautionary_statement": "Remain indoors with windows closed. Avoid physical exertion.",
        }


# Training thread function
def training_thread():
    """Thread function for running the training process"""
    global training_status
    try:
        with training_lock:
            training_status["current_phase"] = "gathering"
            training_status["current_step"] = "Starting data gathering..."
            training_status["progress"] = 0
            training_status["logs"] = []
            training_status["training_complete"] = False
            training_status["best_model"] = None
            training_status["model_metrics"] = {}
            training_status["all_models_data"] = {}
            training_status["feature_importances"] = []

        # Main URL for scraping
        base_url = (
            "https://air-quality.com/country/india/3ffd900b?lang=en&standard=naqi_in"
        )

        # Scrape the data
        df = scrape_all_cities(base_url, max_workers=50)

        if df.empty:
            with training_lock:
                training_status["current_phase"] = "idle"
                training_status["current_step"] = "Error: No data scraped"
                training_status["logs"].append("Error: No data scraped")
            return

        # Save raw data
        df.to_csv("AQI_Raw_Data.csv", index=False)
        with training_lock:
            training_status["logs"].append("Raw data saved to AQI_Raw_Data.csv")

        # Handle missing values
        df_clean = handle_missing_values(df.copy())

        # Save cleaned data
        df_clean.to_csv("AQI_Clean_Data.csv", index=False)
        with training_lock:
            training_status["logs"].append("Cleaned data saved to AQI_Clean_Data.csv")

        # Feature engineering
        with training_lock:
            training_status["current_phase"] = "feature_selection"
            training_status["current_step"] = "Creating additional features..."
            training_status["logs"].append("Creating additional features...")

        df_features, le_state, le_city, le_aqi_type, le_aqi_category = create_features(
            df_clean.copy()
        )

        # Prepare features for modeling
        X, y, feature_names = prepare_features(df_features)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        with training_lock:
            training_status["logs"].append(
                f"Features prepared: {len(feature_names)} features"
            )

        # Feature scaling comparison
        scalers = {
            "StandardScaler": StandardScaler(),
            "MinMaxScaler": MinMaxScaler(),
            "RobustScaler": RobustScaler(),
            "PowerTransformer": PowerTransformer(method="yeo-johnson"),
        }

        # Test each scaler with a simple model
        scaler_results = {}
        for scaler_name, scaler in scalers.items():
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Test with Linear Regression
            lr = LinearRegression()
            lr.fit(X_train_scaled, y_train)
            y_pred = lr.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            scaler_results[scaler_name] = mse

        # Select best scaler
        best_scaler_name = min(scaler_results, key=scaler_results.get)
        best_scaler = scalers[best_scaler_name]

        with training_lock:
            training_status["logs"].append(f"Best scaler selected: {best_scaler_name}")

        # Apply best scaler
        X_train_scaled = best_scaler.fit_transform(X_train)
        X_test_scaled = best_scaler.transform(X_test)

        # Define models to compare (CPU-compatible only)
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(),
            "Lasso Regression": Lasso(),
            "Decision Tree": DecisionTreeRegressor(random_state=42),
            "Random Forest": RandomForestRegressor(random_state=42, n_jobs=-1),
            "Extra Trees": ExtraTreesRegressor(random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
            "AdaBoost": AdaBoostRegressor(random_state=42),
            "XGBoost": xgb.XGBRegressor(random_state=42, n_jobs=-1),
            "LightGBM": lgb.LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1),
            "CatBoost": CatBoostRegressor(
                random_state=42, verbose=False, thread_count=-1
            ),
            "KNN": KNeighborsRegressor(n_jobs=-1),
        }

        # Model training
        with training_lock:
            training_status["current_phase"] = "training"
            training_status["progress"] = 50

        model_results = evaluate_models(
            models, X_train_scaled, X_test_scaled, y_train, y_test
        )

        # DNN model and stacked models removed as they're not CPU-friendly for this application

        # Create results DataFrame
        results_df = pd.DataFrame(
            {
                name: {
                    "MSE": result["MSE"],
                    "MAE": result["MAE"],
                    "R2": result["R2"],
                    "CV_MSE": result["CV_MSE"],
                }
                for name, result in model_results.items()
            }
        ).T

        # Sort by MSE
        results_df = results_df.sort_values("MSE")

        # Get best model
        best_model_name = results_df.index[0]
        best_model = model_results[best_model_name]["model"]
        best_mse = results_df.loc[best_model_name, "MSE"]
        best_mae = results_df.loc[best_model_name, "MAE"]
        best_r2 = results_df.loc[best_model_name, "R2"]

        # Get feature importance for tree-based models
        if hasattr(best_model, "feature_importances_"):
            importance_df = get_feature_importance(best_model, feature_names)
            if importance_df is not None:
                feature_importances = []
                for _, row in importance_df.head(10).iterrows():
                    feature_importances.append(
                        {
                            "feature": row["Feature"],
                            "importance": float(row["Importance"]),
                        }
                    )
                with training_lock:
                    training_status["feature_importances"] = feature_importances

        # Save the best model
        encoders = {
            "state": le_state,
            "city": le_city,
            "aqi_type": le_aqi_type,
            "aqi_category": le_aqi_category,
        }
        save_model(best_model, best_model_name, best_scaler, feature_names, encoders)

        # Update training status
        with training_lock:
            training_status["current_phase"] = "complete"
            training_status["progress"] = 100
            training_status["current_step"] = "Training complete!"
            training_status["training_complete"] = True
            training_status["best_model"] = best_model_name
            training_status["model_metrics"] = {
                "mse": best_mse,
                "mae": best_mae,
                "r2": best_r2,
            }
            # Store all models data for results page
            for name, result in model_results.items():
                training_status["all_models_data"][name] = {
                    "mse": result["MSE"],
                    "mae": result["MAE"],
                    "r2": result["R2"],
                    "model_type": result["model_type"],
                }
            training_status["logs"].append("Training complete!")
            training_status["logs"].append(f"Best model: {best_model_name}")
            training_status["logs"].append(f"MSE: {best_mse:.4f}")
            training_status["logs"].append(f"MAE: {best_mae:.4f}")
            training_status["logs"].append(f"R2: {best_r2:.4f}")

    except Exception as e:
        with training_lock:
            training_status["current_phase"] = "idle"
            training_status["current_step"] = f"Error: {str(e)}"
            training_status["logs"].append(f"Error: {str(e)}")


# Flask routes
@app.route("/")
def index():
    """Redirect to loading page as the default entry point"""
    return render_template("loading.html")


@app.route("/prediction")
def prediction():
    """Render the prediction page"""
    return render_template("index.html")


@app.route("/loading")
def loading():
    """Render the loading page"""
    return render_template("loading.html")


@app.route("/results")
def results():
    """Render the results page"""
    return render_template("results.html")


@app.route("/api/reset_training", methods=["POST"])
def reset_training():
    """Reset the training status to initial state"""
    try:
        with training_lock:
            training_status["current_phase"] = "idle"
            training_status["current_step"] = "Ready to start training"
            training_status["progress"] = 0
            training_status["logs"] = ["System ready. Click 'Start Training' to begin."]
            training_status["training_complete"] = False
            training_status["best_model"] = None
            training_status["model_metrics"] = {}
            training_status["all_models_data"] = {}
            training_status["feature_importances"] = []
        return jsonify(
            {"success": True, "message": "Training status reset successfully"}
        )
    except Exception as e:
        return jsonify(
            {"success": False, "message": f"Error resetting training status: {str(e)}"}
        )


@app.route("/api/model_status")
def model_status():
    """Check if the model is loaded and ready"""
    try:
        # Check if model file exists
        model_files = [
            "saved_models/Extra Trees.pkl",
            "saved_models/Random Forest.pkl",
            "saved_models/XGBoost.pkl",
            "saved_models/Gradient Boosting.pkl",
        ]
        model_found = False
        model_file = None
        for file in model_files:
            if os.path.exists(file):
                model_found = True
                model_file = file
                break

        if not model_found:
            return jsonify(
                {
                    "success": False,
                    "message": "No trained model found. Please train a model first.",
                    "training_complete": training_status["training_complete"],
                    "current_phase": training_status["current_phase"],
                }
            )

        # Get model file info
        file_size = os.path.getsize(model_file)
        file_modified = datetime.fromtimestamp(os.path.getmtime(model_file)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        # Try to determine model type from filename
        model_name = os.path.basename(model_file).split(".")[0]
        model_type = "Unknown"
        if "Extra Trees" in model_name:
            model_type = "Extra Trees Regressor"
        elif "Random Forest" in model_name:
            model_type = "Random Forest Regressor"
        elif "XGBoost" in model_name:
            model_type = "XGBoost Regressor"
        elif "Gradient Boosting" in model_name:
            model_type = "Gradient Boosting Regressor"

        # Determine scaler type
        scaler_type = "Unknown"
        scaler_file = model_file.replace(".pkl", "_scaler.pkl").replace(
            ".h5", "_scaler.pkl"
        )
        if os.path.exists(scaler_file):
            scaler = joblib.load(scaler_file)
            scaler_type = type(scaler).__name__

        return jsonify(
            {
                "success": True,
                "model_name": model_name,
                "model_type": model_type,
                "scaler_type": scaler_type,
                "file_size": file_size,
                "file_modified": file_modified,
                "training_complete": training_status["training_complete"],
                "current_phase": training_status["current_phase"],
            }
        )
    except Exception as e:
        return jsonify(
            {
                "success": False,
                "message": f"Error checking model status: {str(e)}",
                "training_complete": training_status["training_complete"],
                "current_phase": training_status["current_phase"],
            }
        )


@app.route("/api/model_info")
def model_info():
    """Get basic model information"""
    try:
        # Check if model file exists
        model_files = [
            "saved_models/Extra Trees.pkl",
            "saved_models/Random Forest.pkl",
            "saved_models/XGBoost.pkl",
            "saved_models/Gradient Boosting.pkl",
        ]
        model_found = False
        model_file = None
        for file in model_files:
            if os.path.exists(file):
                model_found = True
                model_file = file
                break

        if not model_found:
            return jsonify(
                {
                    "success": False,
                    "message": "No trained model found. Please train a model first.",
                    "training_complete": training_status["training_complete"],
                    "current_phase": training_status["current_phase"],
                }
            )

        # Get model file info
        file_size = os.path.getsize(model_file)
        file_modified = datetime.fromtimestamp(os.path.getmtime(model_file)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )

        return jsonify(
            {
                "success": True,
                "file_size": file_size,
                "file_modified": file_modified,
                "training_complete": training_status["training_complete"],
                "current_phase": training_status["current_phase"],
            }
        )
    except Exception as e:
        return jsonify(
            {
                "success": False,
                "message": f"Error checking model info: {str(e)}",
                "training_complete": training_status["training_complete"],
                "current_phase": training_status["current_phase"],
            }
        )


@app.route("/api/predict", methods=["POST"])
def predict():
    """Predict AQI based on input data"""
    try:
        # Get input data
        input_data = request.json

        # Validate input data
        required_fields = [
            "PM2_5",
            "PM10",
            "O3",
            "SO2",
            "CO",
            "Wind Speed",
            "Humidity",
            "Temp",
        ]
        for field in required_fields:
            if field not in input_data:
                return jsonify(
                    {"success": False, "error": f"Missing required field: {field}"}
                )

        # Find the best model
        model_files = [
            "saved_models/Extra Trees.pkl",
            "saved_models/Random Forest.pkl",
            "saved_models/XGBoost.pkl",
            "saved_models/Gradient Boosting.pkl",
        ]
        model_found = False
        model_file = None
        for file in model_files:
            if os.path.exists(file):
                model_found = True
                model_file = file
                break

        if not model_found:
            return jsonify(
                {
                    "success": False,
                    "error": "No trained model found. Please train a model first.",
                }
            )

        # Load model and related objects
        model_name = os.path.basename(model_file).split(".")[0]
        model, scaler, feature_names, encoders = load_model(model_name)

        # Prepare input data
        prepared_input = {
            "State": "Unknown",  # Default value
            "City": "Unknown",  # Default value
            "AQI Type": "Unknown",  # Default value
            "PM2.5": input_data["PM2_5"],
            "PM10": input_data["PM10"],
            "O3": input_data["O3"],
            "SO2": input_data["SO2"],
            "CO": input_data["CO"],
            "NO2": 0,  # Default value
            "Wind Speed": input_data["Wind Speed"],
            "Humidity": input_data["Humidity"],
            "Temperature": input_data["Temp"],
            "AQI": 0,  # Default value
        }

        # Make prediction
        prediction = predict_aqi(prepared_input, model, scaler, feature_names, encoders)

        # Get AQI type and color
        aqi_type = get_aqi_type(prediction)
        aqi_color = get_aqi_color(prediction)

        # Get AQI advice
        aqi_advice = get_aqi_advice(prediction)

        return jsonify(
            {
                "success": True,
                "prediction": round(float(prediction), 2),
                "aqi_type": aqi_type,
                "color": aqi_color,
                "model_name": model_name,
                "advice": aqi_advice,
            }
        )
    except Exception as e:
        return jsonify(
            {"success": False, "error": f"Error making prediction: {str(e)}"}
        )


@app.route("/api/dataset_cities")
def dataset_cities():
    """Get cities and AQI data from the dataset"""
    try:
        # Check if cleaned data file exists
        if os.path.exists("AQI_Clean_Data.csv"):
            df = pd.read_csv("AQI_Clean_Data.csv")
        else:
            # If cleaned data doesn't exist, try raw data
            if os.path.exists("AQI_Raw_Data.csv"):
                df = pd.read_csv("AQI_Raw_Data.csv")
            else:
                return jsonify(
                    {
                        "success": False,
                        "error": "No dataset found. Please scrape data first.",
                    }
                )

        # Convert to list of dictionaries
        cities = df.to_dict("records")
        return jsonify({"success": True, "cities": cities})
    except Exception as e:
        return jsonify({"success": False, "error": f"Error loading dataset: {str(e)}"})


@app.route("/api/city_data/<city_name>")
def get_city_data(city_name):
    """Get specific city data for auto-filling prediction form"""
    try:
        if os.path.exists("AQI_Clean_Data.csv"):
            df = pd.read_csv("AQI_Clean_Data.csv")
        else:
            if os.path.exists("AQI_Raw_Data.csv"):
                df = pd.read_csv("AQI_Raw_Data.csv")
            else:
                return jsonify({"success": False, "error": "No dataset found."})

        # Find the city data
        city_data = df[df["City"].str.contains(city_name, case=False, na=False)]
        if city_data.empty:
            return jsonify({"success": False, "error": "City not found."})

        # Convert to dictionary
        city_dict = city_data.iloc[0].to_dict()
        return jsonify({"success": True, "data": city_dict})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/start_phase", methods=["POST"])
def start_phase():
    """Start the training process"""
    try:
        # Check if training is already in progress
        if training_status["current_phase"] not in ["idle", "complete"]:
            return jsonify(
                {
                    "success": False,
                    "message": f"Training already in progress: {training_status['current_phase']}",
                }
            )

        # Start training in a separate thread
        thread = threading.Thread(target=training_thread)
        thread.daemon = True
        thread.start()

        return jsonify({"success": True, "message": "Training started"})
    except Exception as e:
        return jsonify(
            {"success": False, "message": f"Error starting training: {str(e)}"}
        )


@app.route("/api/training_status")
def training_status_api():
    """Get the current training status"""
    with training_lock:
        return jsonify(training_status)


@app.route("/api/refresh_data", methods=["POST"])
def refresh_data():
    """Refresh the AQI data by scraping"""
    try:
        # Check if training is already in progress
        if training_status["current_phase"] not in ["idle", "complete"]:
            return jsonify(
                {
                    "success": False,
                    "message": f"Training already in progress: {training_status['current_phase']}",
                }
            )

        # Start data gathering in a separate thread
        def gather_data_only():
            global training_status
            try:
                with training_lock:
                    training_status["current_phase"] = "gathering"
                    training_status["current_step"] = "Starting data extraction..."
                    training_status["progress"] = 0
                    training_status["logs"] = ["Starting data extraction..."]

                # Main URL for scraping
                base_url = "https://air-quality.com/country/india/3ffd900b?lang=en&standard=naqi_in"

                # Scrape the data
                df = scrape_all_cities(base_url, max_workers=50)

                if df.empty:
                    with training_lock:
                        training_status["current_phase"] = "idle"
                        training_status["current_step"] = "Error: No data scraped"
                        training_status["logs"].append("Error: No data scraped")
                    return

                # Save raw data
                df.to_csv("AQI_Raw_Data.csv", index=False)

                # Handle missing values
                df_clean = handle_missing_values(df.copy())

                # Save cleaned data
                df_clean.to_csv("AQI_Clean_Data.csv", index=False)

                with training_lock:
                    training_status["current_phase"] = "complete"
                    training_status["progress"] = 100
                    training_status["current_step"] = "Data refresh complete!"
                    training_status["logs"].append("Data refresh complete!")
            except Exception as e:
                with training_lock:
                    training_status["current_phase"] = "idle"
                    training_status["current_step"] = f"Error: {str(e)}"
                    training_status["logs"].append(f"Error: {str(e)}")

        thread = threading.Thread(target=gather_data_only)
        thread.daemon = True
        thread.start()

        return jsonify({"success": True, "message": "Data refresh started"})
    except Exception as e:
        return jsonify(
            {"success": False, "message": f"Error refreshing data: {str(e)}"}
        )


@app.route("/api/model_comparison_chart")
def model_comparison_chart():
    """Generate and return a model comparison chart"""
    try:
        # Check if we have model results
        if not training_status["all_models_data"]:
            return jsonify(
                {
                    "success": False,
                    "error": "No model results available. Please train models first.",
                }
            )

        # Create a comparison chart
        models_data = training_status["all_models_data"]
        model_names = list(models_data.keys())
        mse_values = [models_data[name]["mse"] for name in model_names]
        mae_values = [models_data[name]["mae"] for name in model_names]
        r2_values = [models_data[name]["r2"] for name in model_names]

        # Create figure
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "MSE Comparison",
                "MAE Comparison",
                "R² Comparison",
                "Model Overview",
            ),
            specs=[
                [{"secondary_y": False}, {"secondary_y": False}],
                [{"secondary_y": False}, {"type": "table"}],
            ],
        )

        # MSE
        fig.add_trace(
            go.Bar(x=model_names, y=mse_values, marker_color="lightblue", name="MSE"),
            row=1,
            col=1,
        )

        # MAE
        fig.add_trace(
            go.Bar(x=model_names, y=mae_values, marker_color="lightgreen", name="MAE"),
            row=1,
            col=2,
        )

        # R2
        fig.add_trace(
            go.Bar(x=model_names, y=r2_values, marker_color="lightcoral", name="R²"),
            row=2,
            col=1,
        )

        # Table
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Model", "MSE", "MAE", "R²"],
                    fill_color="lightgrey",
                    align="left",
                ),
                cells=dict(
                    values=[
                        model_names,
                        [f"{val:.4f}" for val in mse_values],
                        [f"{val:.4f}" for val in mae_values],
                        [f"{val:.4f}" for val in r2_values],
                    ],
                    fill_color="white",
                    align="left",
                ),
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            height=800, showlegend=False, title_text="Model Performance Comparison"
        )

        # Convert to image
        img_bytes = fig.to_image(format="png")
        img_base64 = img_bytes.decode("utf-8").replace("'", "")

        return jsonify({"success": True, "chart_data": img_base64})
    except Exception as e:
        return jsonify({"success": False, "error": f"Error generating chart: {str(e)}"})


# Main entry point
if __name__ == "__main__":
    # Initialize training status
    with training_lock:
        training_status["current_phase"] = "idle"
        training_status["current_step"] = "Ready to start training"
        training_status["progress"] = 0
        training_status["logs"] = ["System ready. Click 'Gather Data' to begin."]
        training_status["training_complete"] = False

    # Check if we have existing data
    if os.path.exists("AQI_Clean_Data.csv"):
        print("Found existing AQI data")
    else:
        print("No existing AQI data found")

    # Check if we have a trained model
    model_files = [
        "saved_models/Extra Trees.pkl",
        "saved_models/Random Forest.pkl",
        "saved_models/XGBoost.pkl",
        "saved_models/Gradient Boosting.pkl",
    ]
    model_found = False
    for file in model_files:
        if os.path.exists(file):
            model_found = True
            print(f"Found existing model: {file}")
            with training_lock:
                training_status["training_complete"] = True
            break

    if not model_found:
        print("No existing model found")

    # Start the Flask app
    app.run(debug=True, host="0.0.0.0", port=5000)
