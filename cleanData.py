import pandas as pd
from pathlib import Path

# =========================
# 1. Dateipfade anpassen
# =========================

input_file = Path("listings.csv")          # oder "listings.csv.gz"
output_file = Path("listings_cleaned.csv") # neue Datei, alte bleibt erhalten


# =========================
# 2. Datei laden
# =========================

df = pd.read_csv(
    input_file,
    compression="infer",
    low_memory=False
)

print("Original Shape:", df.shape)


# =========================
# 3. Boolean-Spalten t/f -> 1/0
# =========================

boolean_cols = [
    "host_is_superhost",
    "host_has_profile_pic",
    "host_identity_verified",
    "has_availability",
    "instant_bookable"
]

for col in boolean_cols:
    if col in df.columns:
        df[col] = df[col].map({
            "t": 1,
            "f": 0,
            True: 1,
            False: 0
        }).astype("Int64")


# =========================
# 4. Prozent-Spalten bereinigen
# Beispiel: "96%" -> 0.96
# In der CSV-Ausgabe wird daraus wegen decimal="," dann 0,96
# =========================

percent_cols = [
    "host_response_rate",
    "host_acceptance_rate"
]

for col in percent_cols:
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("%", "", regex=False)
            .str.replace(",", ".", regex=False)
            .replace(["nan", "N/A", "None", ""], pd.NA)
        )
        df[col] = pd.to_numeric(df[col], errors="coerce") / 100


# =========================
# 5. Preis-Spalte bereinigen
# Beispiel: "$1,234.00" -> 1234.00
# In der CSV-Ausgabe wird daraus 1234,00
# =========================

if "price" in df.columns:
    df["price"] = (
        df["price"]
        .astype(str)
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .replace(["nan", "N/A", "None", ""], pd.NA)
    )
    df["price"] = pd.to_numeric(df["price"], errors="coerce")


# =========================
# 6. Datums-Spalten vereinheitlichen
# Format bleibt: YYYY-MM-DD
# =========================

date_cols = [
    "last_scraped",
    "host_since",
    "calendar_last_scraped",
    "first_review",
    "last_review"
]

for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce").dt.strftime("%Y-%m-%d")


# =========================
# 7. Komplett leere Spalten entfernen
# =========================

df = df.dropna(axis=1, how="all")


# =========================
# 8. Neue CSV speichern
# sep=";"       -> Semikolon als Trennzeichen
# decimal=","   -> Dezimalkomma
# =========================

df.to_csv(
    output_file,
    sep=";",
    decimal=",",
    index=False,
    encoding="utf-8-sig"
)

print("Bereinigte Datei gespeichert als:", output_file)
print("Cleaned Shape:", df.shape)