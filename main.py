"""
main.py — FastAPI backend for AI Field Force Intelligence
=========================================================

CORRECTED VERSION
-----------------
This version is aligned with build_master_table.py training logic.

Key fixes:
1. Recency feature now includes territory_id (training-serving parity)
2. Inventory fallback now returns NaN like training pipeline
3. Digital funnel logic aligned with training
4. Feature engineering mirrors training semantics
5. Safer label encoding
6. Clear separation between raw features and transformed features

Run:
    uvicorn main:app --reload --port 8000
"""

import json
import pickle
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path

warnings.filterwarnings("ignore")


# ============================================================
# APP INIT
# ============================================================

app = FastAPI(
    title="AI Field Force Intelligence API",
    description="Route optimization and next-best-action engine",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD MODELS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"
DATA_DIR  = BASE_DIR / "data"

clf = pickle.load(open(MODEL_DIR / "model_classifier.pkl", "rb"))

le = pickle.load(open(MODEL_DIR / "label_encoders.pkl", "rb"))

FEATURE_COLS = pickle.load(open(MODEL_DIR / "feature_cols.pkl", "rb"))

# clf = pickle.load(open("model_classifier.pkl", "rb"))
# le = pickle.load(open("label_encoders.pkl", "rb"))
# FEATURE_COLS = pickle.load(open("feature_cols.pkl", "rb"))


# ============================================================
# LOAD RAW TABLES
# ============================================================



print("Loading raw tables...")

retailers = pd.read_csv(DATA_DIR / "retailers.csv")

reps = pd.read_csv(DATA_DIR / "reps_territory.csv")

pos = pd.read_csv(
    DATA_DIR / "retailer_pos.csv",
    parse_dates=["transaction_date"]
)

inventory = pd.read_csv(
    DATA_DIR / "retailer_inventory_weekly.csv",
    parse_dates=["week_end_date"]
)

growers = pd.read_csv(DATA_DIR / "growers.csv")

whatsapp = pd.read_csv(
    DATA_DIR / "whatsapp_campaign.csv",
    parse_dates=["message_sent_date"]
)

digital = pd.read_csv(
    DATA_DIR / "digital_funnel_weekly.csv",
    parse_dates=["week_start_date"]
)

visit_log = pd.read_csv(
    DATA_DIR / "retailer_visit_log.csv",
    parse_dates=["visit_date"]
)


# retailers = pd.read_csv("retailers.csv")
# reps = pd.read_csv("reps_territory.csv")

# pos = pd.read_csv(
#     "retailer_pos.csv",
#     parse_dates=["transaction_date"]
# )

# inventory = pd.read_csv(
#     "retailer_inventory_weekly.csv",
#     parse_dates=["week_end_date"]
# )

# growers = pd.read_csv("growers.csv")

# whatsapp = pd.read_csv(
#     "whatsapp_campaign.csv",
#     parse_dates=["message_sent_date"]
# )

# digital = pd.read_csv(
#     "digital_funnel_weekly.csv",
#     parse_dates=["week_start_date"]
# )

# visit_log = pd.read_csv(
#     "retailer_visit_log.csv",
#     parse_dates=["visit_date"]
# )


# ============================================================
# PRECOMPUTE LOOKUPS
# ============================================================

print("Precomputing lookup structures...")

pos["revenue"] = pos["sku_qty"] * pos["sku_price"]

pos_by_retailer = {
    rid: grp.sort_values("transaction_date")
    for rid, grp in pos.groupby("retailer_id")
}

inv_by_retailer = {
    rid: grp.sort_values("week_end_date")
    for rid, grp in inventory.groupby("retailer_id")
}

grower_agg = (
    growers.groupby("tehsil")
    .agg(
        grower_count=("grower_id", "count"),
        grower_scan_rate=("product_scan", "mean"),
        grower_campaign_rate=("offline_campaign_attended", "mean"),
        avg_farm_size=("grower_farm_size", "mean"),
    )
    .reset_index()
)

wa_with_tehsil = whatsapp.merge(
    growers[["grower_id", "tehsil"]],
    on="grower_id",
    how="left"
)

wa_agg = (
    wa_with_tehsil.groupby("tehsil")
    .agg(
        wa_click_rate=("clicked_status", "mean"),
        wa_open_rate=("opened_status", "mean"),
        wa_delivered_rate=("delivered_status", "mean"),
    )
    .reset_index()
)

digital_sorted = digital.sort_values("week_start_date")

print("✓ Models and lookup layers loaded")


# ============================================================
# SAFE ENCODING
# ============================================================

def safe_transform(encoder, value, default=0):
    if value in encoder.classes_:
        return encoder.transform([value])[0]
    return default


# ============================================================
# FEATURE FUNCTIONS
# ============================================================

CROP_ORDER = [
    "pre_sowing",
    "establishment",
    "tillering",
    "flowering",
    "grain_fill",
    "near_harvest",
]


def get_crop_stage(visit_date):

    vd = pd.Timestamp(visit_date)

    if vd < pd.Timestamp("2025-11-15"):
        return "pre_sowing"

    elif vd < pd.Timestamp("2026-01-01"):
        return "establishment"

    elif vd < pd.Timestamp("2026-01-20"):
        return "tillering"

    elif vd < pd.Timestamp("2026-02-25"):
        return "flowering"

    elif vd < pd.Timestamp("2026-03-20"):
        return "grain_fill"

    else:
        return "near_harvest"


# ============================================================
# POS FEATURES
# ============================================================

def get_pos_features(retailer_id, visit_date):

    if retailer_id not in pos_by_retailer:
        return {
            "pos_revenue_30d": 0.0,
            "pos_txn_30d": 0,
            "pos_units_30d": 0.0,
            "pos_unique_skus": 0,
        }

    df = pos_by_retailer[retailer_id]

    window = df[
        (df["transaction_date"] < visit_date)
        & (df["transaction_date"] >= visit_date - timedelta(days=30))
    ]

    return {
        "pos_revenue_30d": window["revenue"].sum(),
        "pos_txn_30d": len(window),
        "pos_units_30d": window["sku_qty"].sum(),
        "pos_unique_skus": window["sku_id"].nunique(),
    }


# ============================================================
# INVENTORY FEATURES
# ============================================================

def get_inv_features(retailer_id, visit_date):

    if retailer_id not in inv_by_retailer:
        return {
            "avg_stock": np.nan,
            "min_stock": np.nan,
            "stock_out_skus": np.nan,
            "n_skus_stocked": 0,
        }

    df = inv_by_retailer[retailer_id]

    before_visit = df[df["week_end_date"] <= visit_date]

    if len(before_visit) == 0:
        return {
            "avg_stock": np.nan,
            "min_stock": np.nan,
            "stock_out_skus": np.nan,
            "n_skus_stocked": 0,
        }

    latest_per_sku = (
        before_visit
        .sort_values("week_end_date")
        .groupby("sku_id")
        .last()
        .reset_index()
    )

    return {
        "avg_stock": latest_per_sku["sku_qty"].mean(),
        "min_stock": latest_per_sku["sku_qty"].min(),
        "stock_out_skus": (latest_per_sku["sku_qty"] == 0).sum(),
        "n_skus_stocked": len(latest_per_sku),
    }


# ============================================================
# RECENCY FEATURE
# FIXED — NOW MATCHES TRAINING
# ============================================================

def get_recency_feature(territory_id, tehsil, visit_date):

    past_visits = visit_log[
        (visit_log["territory_id"] == territory_id)
        & (visit_log["visit_tehsil"] == tehsil)
        & (visit_log["visit_date"] < visit_date)
    ]

    if len(past_visits) == 0:
        return {
            "days_since_last_visit": 999
        }

    return {
        "days_since_last_visit":
            (visit_date - past_visits["visit_date"].max()).days
    }


# ============================================================
# GROWER FEATURES
# ============================================================

def get_grower_features(tehsil):

    row = grower_agg[grower_agg["tehsil"] == tehsil]

    if len(row) == 0:
        return {
            "grower_count": np.nan,
            "grower_scan_rate": np.nan,
            "grower_campaign_rate": np.nan,
            "avg_farm_size": np.nan,
        }

    r = row.iloc[0]

    return {
        "grower_count": r["grower_count"],
        "grower_scan_rate": r["grower_scan_rate"],
        "grower_campaign_rate": r["grower_campaign_rate"],
        "avg_farm_size": r["avg_farm_size"],
    }


# ============================================================
# WHATSAPP FEATURES
# ============================================================

def get_wa_features(tehsil):

    row = wa_agg[wa_agg["tehsil"] == tehsil]

    if len(row) == 0:
        return {
            "wa_click_rate": np.nan,
            "wa_open_rate": np.nan,
            "wa_delivered_rate": np.nan,
        }

    r = row.iloc[0]

    return {
        "wa_click_rate": r["wa_click_rate"],
        "wa_open_rate": r["wa_open_rate"],
        "wa_delivered_rate": r["wa_delivered_rate"],
    }


# ============================================================
# DIGITAL FUNNEL FEATURES
# ============================================================

def get_digital_features(visit_date, crop="wheat"):

    cutoff = visit_date - timedelta(weeks=4)

    window = digital_sorted[
        (digital_sorted["campaign_crop"] == crop)
        & (digital_sorted["week_start_date"] <= visit_date)
        & (digital_sorted["week_start_date"] >= cutoff)
    ]

    return {
        "digital_impressions_4w":
            window["social_post_impression"].sum(),

        "digital_leads_4w":
            window["lead_form_submission"].sum(),
    }


# ============================================================
# FEATURE ENCODING
# ============================================================

def encode_features(
    state,
    visit_type,
    product_recommended,
    crop_stage
):

    return {
        "state_enc":
            safe_transform(le["state"], state),

        "visit_type_enc":
            safe_transform(le["visit_type"], visit_type),

        "product_recommended_enc":
            safe_transform(
                le["product_recommended"],
                product_recommended
            ),

        "crop_stage_enc":
            CROP_ORDER.index(crop_stage),
    }


# ============================================================
# MASTER FEATURE ROW BUILDER
# ============================================================

def build_feature_row(
    retailer_id,
    territory_id,
    tehsil,
    state,
    visit_date,
    visit_type,
    product_recommended,
):

    crop_stage = get_crop_stage(visit_date)

    row = {}

    # categorical encodings
    row.update(
        encode_features(
            state,
            visit_type,
            product_recommended,
            crop_stage,
        )
    )

    # inventory
    row.update(
        get_inv_features(
            retailer_id,
            visit_date
        )
    )

    # pos
    row.update(
        get_pos_features(
            retailer_id,
            visit_date
        )
    )

    # recency
    row.update(
        get_recency_feature(
            territory_id,
            tehsil,
            visit_date
        )
    )

    # temporal
    row["visit_month"] = visit_date.month
    row["visit_dow"] = visit_date.dayofweek

    # grower
    row.update(
        get_grower_features(tehsil)
    )

    # whatsapp
    row.update(
        get_wa_features(tehsil)
    )

    # digital
    row.update(
        get_digital_features(
            visit_date,
            crop="wheat"
        )
    )

    # ========================================================
    # OPTIONAL LOG TRANSFORMS
    # ONLY KEEP IF TRAINING USED THEM
    # ========================================================

    if "pos_revenue_30d_log" in FEATURE_COLS:
        row["pos_revenue_30d_log"] = np.log1p(
            row["pos_revenue_30d"]
        )

    if "pos_units_30d_log" in FEATURE_COLS:
        row["pos_units_30d_log"] = np.log1p(
            row["pos_units_30d"]
        )

    if "days_since_last_visit_log" in FEATURE_COLS:
        row["days_since_last_visit_log"] = np.log1p(
            row["days_since_last_visit"]
        )

    return pd.DataFrame([row])[FEATURE_COLS]


# ============================================================
# REASON BUILDER
# ============================================================

def build_reason(features):

    reasons = []

    if pd.notna(features.get("avg_stock")):
        if features["avg_stock"] < 40:
            reasons.append(
                f"Low stock ({features['avg_stock']:.0f} units)"
            )

    if features.get("pos_revenue_30d", 0) > 80000:
        reasons.append(
            f"High recent POS sales"
        )

    if features.get("days_since_last_visit", 0) > 20:
        reasons.append(
            f"No recent rep engagement"
        )

    if features.get("wa_click_rate", 0) > 0.10:
        reasons.append(
            "Strong WhatsApp engagement"
        )

    crop = features.get("crop_stage", "")

    if crop in ["flowering", "grain_fill"]:
        reasons.append(
            f"{crop} stage application window active"
        )

    if not reasons:
        reasons.append(
            "High contextual conversion likelihood"
        )

    return " | ".join(reasons)


# ============================================================
# REQUEST SCHEMAS
# ============================================================

class PredictRequest(BaseModel):
    retailer_id: str
    tehsil: str
    state: str
    territory_id: str
    visit_date: str
    visit_type: str = "retailer meeting"
    product_recommended: str = "Tilt 250 EC"


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/")
def health():

    return {
        "status": "running",
        "version": "2.0.0",
        "model": "AI Field Force Intelligence"
    }


# ============================================================
# SINGLE PREDICTION
# ============================================================

@app.post("/predict")
def predict(req: PredictRequest):

    visit_date = pd.Timestamp(req.visit_date)

    X = build_feature_row(
        retailer_id=req.retailer_id,
        territory_id=req.territory_id,
        tehsil=req.tehsil,
        state=req.state,
        visit_date=visit_date,
        visit_type=req.visit_type,
        product_recommended=req.product_recommended,
    )

    p = float(clf.predict_proba(X)[0, 1])

    return {
        "retailer_id": req.retailer_id,
        "territory_id": req.territory_id,
        "tehsil": req.tehsil,
        "visit_date": req.visit_date,
        "p_conversion": round(p, 4),
        "priority_score": round(p * 100, 1),
        "crop_stage": get_crop_stage(visit_date),
    }


# ============================================================
# ROUTE RECOMMENDATION
# ============================================================

@app.get("/route")
def get_route(
    rep_id: str = Query(...),
    date: str = Query(...),
    top_n: int = Query(8)
):

    rep_row = reps[reps["rep_id"] == rep_id]

    if len(rep_row) == 0:
        raise HTTPException(
            status_code=404,
            detail=f"Rep {rep_id} not found"
        )

    rep_data = rep_row.iloc[0]

    territory_id = rep_data["territory_id"]

    allowed_tehsils = json.loads(
        rep_data["tehsil_list"]
    )

    my_retailers = retailers[
        retailers["tehsil"].isin(allowed_tehsils)
    ].copy()

    visit_date = pd.Timestamp(date)

    scored = []

    for _, r in my_retailers.iterrows():

        try:

            X = build_feature_row(
                retailer_id=r["retailer_id"],
                territory_id=territory_id,
                tehsil=r["tehsil"],
                state=r["state"],
                visit_date=visit_date,
                visit_type="retailer meeting",
                product_recommended="Tilt 250 EC",
            )

            p = float(clf.predict_proba(X)[0, 1])

            raw_pos = get_pos_features(
                r["retailer_id"],
                visit_date
            )

            raw_inv = get_inv_features(
                r["retailer_id"],
                visit_date
            )

            raw_rec = get_recency_feature(
                territory_id,
                r["tehsil"],
                visit_date
            )

            raw_wa = get_wa_features(
                r["tehsil"]
            )

            reason_features = {
                **raw_pos,
                **raw_inv,
                **raw_rec,
                **raw_wa,
                "crop_stage": get_crop_stage(visit_date),
            }

            scored.append({
                "retailer_id": r["retailer_id"],
                "tehsil": r["tehsil"],
                "district": r["district"],
                "state": r["state"],
                "p_conversion": round(p, 4),
                "priority_score": round(p * 100, 1),
                "reason": build_reason(reason_features),
            })

        except Exception as e:
            print(f"Scoring error: {e}")
            continue

    ranked = sorted(
        scored,
        key=lambda x: x["p_conversion"],
        reverse=True
    )[:top_n]

    for i, r in enumerate(ranked):
        r["rank"] = i + 1

    return {
        "rep_id": rep_id,
        "territory_id": territory_id,
        "date": date,
        "total_scored": len(scored),
        "route": ranked,
    }