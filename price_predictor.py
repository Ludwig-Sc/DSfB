"""
Airbnb London Price Predictor - Gradient Boosting (v2)
========================================================
Merges Inside Airbnb geo, reviews, booking-rules, bedrooms/room-type,
and amenities CSVs, trains a Gradient Boosting Regressor on price,
and exposes predict_price() for new listings.

Data files expected in the same folder as this script:
    london_geo_features.csv
    06_reviews_ratings-2.csv
    05_booking_rules_availability-3.csv
    listings_final_bedrooms-2.csv
    amenities_reduced_grouped.csv

Usage:
    python price_predictor.py
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TODAY = pd.Timestamp("2026-07-05")

COMMA_COLS_REVIEWS = [
    "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
    "review_scores_checkin", "review_scores_communication", "review_scores_location",
    "review_scores_value", "reviews_per_month",
]
COMMA_COLS_BOOKING = [
    "minimum_minimum_nights", "maximum_minimum_nights", "minimum_maximum_nights",
    "maximum_maximum_nights", "minimum_nights_avg_ntm", "maximum_nights_avg_ntm",
]
BRIDGE_KEY_COLS = ["price", "neighbourhood_cleansed", "number_of_reviews", "review_scores_rating"]
LISTING_FEATURE_COLS = ["bedrooms", "is_studio", "bedrooms_log", "accommodates", "beds", "room_type"]

param_dist = {
    'max_iter': [200, 300, 400, 500],
    'max_depth': [4, 5, 6, 7, 8, None],
    'learning_rate': [0.03, 0.05, 0.08, 0.1, 0.15],
    'min_samples_leaf': [5, 10, 20, 30],
    'l2_regularization': [0.0, 0.1, 0.5, 1.0],
    'max_leaf_nodes': [15, 31, 63, 127],
}

def to_numeric_comma(series):
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def load_all(script_dir):
    geo = pd.read_csv(os.path.join(script_dir, "london_geo_features.csv"), sep=";")
    reviews = pd.read_csv(os.path.join(script_dir, "06_reviews_ratings.csv"), sep=";")
    booking = pd.read_csv(os.path.join(script_dir, "05_booking_rules_availability.csv"), sep=";")
    listings = pd.read_csv(os.path.join(script_dir, "listings_final_bedrooms.csv"), sep=";")
    amenities = pd.read_csv(os.path.join(script_dir, "amenities_reduced_grouped.csv"), sep=";")
    return geo, reviews, booking, listings, amenities


def clean_inputs(geo, reviews, booking, listings, amenities):
    for c in COMMA_COLS_REVIEWS:
        reviews[c] = to_numeric_comma(reviews[c])
    for c in COMMA_COLS_BOOKING:
        booking[c] = to_numeric_comma(booking[c])
    booking["has_availability"] = to_numeric_comma(booking["has_availability"])

    amenities["price"] = to_numeric_comma(amenities["price"])
    listings["beds"] = to_numeric_comma(listings["beds"])
    listings["review_scores_rating"] = to_numeric_comma(listings["review_scores_rating"])
    return geo, reviews, booking, listings, amenities


def merge_id_based(geo, reviews, booking, amenities):
    """Merge the three files that share a real 'id' column."""
    full = (
        geo.merge(reviews, on="id", how="left")
           .merge(booking, on="id", how="left")
           .merge(amenities.drop(columns=["price"]), on="id", how="left")
    )
    return full


def bridge_listings_features(full, listings):
    """listings_final_bedrooms-2.csv has no id column, so we recover the
    link via a composite key (price, neighbourhood, review count, rating)."""
    full_key = full.dropna(subset=BRIDGE_KEY_COLS).drop_duplicates(subset=BRIDGE_KEY_COLS, keep="first")
    listings_key = listings.dropna(subset=BRIDGE_KEY_COLS).drop_duplicates(subset=BRIDGE_KEY_COLS, keep="first")

    bridge = (
        full_key[["id"] + BRIDGE_KEY_COLS]
        .merge(listings_key[BRIDGE_KEY_COLS + LISTING_FEATURE_COLS], on=BRIDGE_KEY_COLS, how="inner")
        .drop_duplicates(subset="id", keep="first")
    )
    return full.merge(bridge[["id"] + LISTING_FEATURE_COLS], on="id", how="left")


def engineer_features(df):
    df["first_review"] = pd.to_datetime(df["first_review"], errors="coerce")
    df["last_review"] = pd.to_datetime(df["last_review"], errors="coerce")
    df["host_experience_days"] = (TODAY - df["first_review"]).dt.days
    df["days_since_last_review"] = (TODAY - df["last_review"]).dt.days
    df = df.drop(columns=["neighbourhood", "first_review", "last_review"])
    return df


def encode_and_impute(df):
    le_neigh = LabelEncoder()
    df["neighbourhood_cleansed"] = le_neigh.fit_transform(df["neighbourhood_cleansed"])

    df["room_type"] = df["room_type"].fillna("Unknown")
    le_room = LabelEncoder()
    df["room_type"] = le_room.fit_transform(df["room_type"])

    for c in df.columns:
        if c in ("id", "price"):
            continue
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())

    return df, le_neigh, le_room


def train_model(X_train, y_train, n_estimators=100, max_depth=4, learning_rate=0.1):
    hgb = HistGradientBoostingRegressor(
        max_iter=1000,
        min_samples_leaf=10,
        max_depth=max_depth,
        l2_regularization=0.1,
        learning_rate=learning_rate,
        random_state=42,
    )
    hgb.fit(X_train, y_train)
    return hgb


def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    r2 = r2_score(y_test, pred)
    print(f"MAE: {mae:.2f}  RMSE: {rmse:.2f}  R2: {r2:.4f}")

    perm = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1)
    importances = pd.Series(perm.importances_mean, index=X_test.columns).sort_values(ascending=False)
    print("\nTop 15 feature importances (permutation-based):")
    print(importances.head(15))

    return pred, mae, rmse, r2


def predict_price(listing: dict, model, le_neigh, le_room, feature_columns, medians) -> float:
    """Predict nightly price (GBP) for a new listing dict.
    Missing keys are auto-filled with training-set medians."""
    row = {}
    for col in feature_columns:
        row[col] = listing[col] if col in listing else medians[col]

    if isinstance(row["neighbourhood_cleansed"], str):
        row["neighbourhood_cleansed"] = le_neigh.transform([row["neighbourhood_cleansed"]])[0]
    if isinstance(row["room_type"], str):
        row["room_type"] = le_room.transform([row["room_type"]])[0]

    X_new = pd.DataFrame([row])[feature_columns]
    return float(model.predict(X_new)[0])


def main():
    geo, reviews, booking, listings, amenities = load_all(SCRIPT_DIR)
    geo, reviews, booking, listings, amenities = clean_inputs(geo, reviews, booking, listings, amenities)

    full = merge_id_based(geo, reviews, booking, amenities)
    full2 = bridge_listings_features(full, listings)

    df = engineer_features(full2)
    df, le_neigh, le_room = encode_and_impute(df)

    X = df.drop(columns=["id", "price"])
    y = df["price"]
    feature_columns = X.columns.tolist()
    medians = X.median().to_dict()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)
    evaluate(model, X_test, y_test)

    search = RandomizedSearchCV(
        HistGradientBoostingRegressor(random_state=42),
        param_distributions=param_dist,
        n_iter=30,
        cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    search.fit(X_train, y_train)
    print("Best params:", search.best_params_)
    print("Best CV MAE:", -search.best_score_)

    best_model = search.best_estimator_

    with open("gb_price_model_v3.pkl", "wb") as f:
        pickle.dump({
            "model": best_model,
            "le_neigh": le_neigh,
            "le_room": le_room,
            "feature_columns": feature_columns,
            "medians": medians,
        }, f)
    print("\nModel saved to gb_price_model_v3.pkl")



    example = {
        "neighbourhood_cleansed": "Camden",
        "latitude": 51.539,
        "longitude": -0.142,
        "dist_center_km": 3.2,
        "neighbourhood_avg_price": 145.0,
        "number_of_reviews": 25,
        "review_scores_rating": 4.85,
        "availability_365": 180,
        "minimum_nights": 2,
        "bedrooms": 1,
        "accommodates": 2,
        "room_type": "Entire home/apt",
        "has_wifi": 1,
        "has_kitchen": 1,
    }
    price = predict_price(example, best_model, le_neigh, le_room, feature_columns, medians)
    print(f"\nExample prediction (Camden, 1BR entire home): £{price:.2f}")


if __name__ == "__main__":
    main()