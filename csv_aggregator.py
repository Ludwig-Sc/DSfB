import pandas as pd

# --- Load --------------------------------------------------
listings = pd.read_csv("listings_cleaned.csv")
reviews  = pd.read_csv("reviews.csv")
calendar = pd.read_csv("calendar.csv")

listings["id"] = listings["id"].astype(str)
reviews["listing_id"]  = reviews["listing_id"].astype(str)
calendar["listing_id"] = calendar["listing_id"].astype(str)

# -- 1. Review summary per listing --------------------------

review_summary = (
    reviews
    .groupby("listing_id")
    .agg(
        total_reviews      = ("id",       "count"),
        first_review_date  = ("date",     "min"),
        last_review_date   = ("date",     "max"),
        avg_comment_length = ("comments", lambda x: x.dropna().str.split().str.len().mean()),
    )
    .reset_index()
    .rename(columns={"listing_id": "id"})
)

listings_reviews = listings.merge(review_summary, on="id", how="left")
listings_reviews.to_csv("listings_with_review_summary.csv", index=False)
print(f"listings_reviews shape: {listings_reviews.shape}")

# -- 2. Calendar summary per listing ---------------------------

calendar["price_clean"] = (
    calendar["price"]
    .astype(str)
    .str.replace(r"[\$,]", "", regex=True)
    .pipe(pd.to_numeric, errors="coerce")
)

calendar_summary = (
    calendar
    .groupby("listing_id")
    .agg(
        total_calendar_days  = ("available",   "count"),
        days_available       = ("available",   lambda x: (x == "t").sum()),
        days_booked          = ("available",   lambda x: (x == "f").sum()),
        avg_price            = ("price_clean", "mean"),
        min_price            = ("price_clean", "min"),
        max_price            = ("price_clean", "max"),
        earliest_date        = ("date",        "min"),
        latest_date          = ("date",        "max"),
    )
    .reset_index()
    .rename(columns={"listing_id": "id"})
)
calendar_summary["occupancy_rate_pct"] = (
    calendar_summary["days_booked"] / calendar_summary["total_calendar_days"] * 100
).round(2)

listings_calendar = listings.merge(calendar_summary, on="id", how="left")
listings_calendar.to_csv("listings_with_calendar_summary.csv", index=False)
print(f"listings_calendar shape: {listings_calendar.shape}")
