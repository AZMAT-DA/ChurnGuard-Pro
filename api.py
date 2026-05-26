# api.py — ChurnGuard Pro REST API
# Run with: uvicorn api:app --reload --port 8000

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import json
import io
from industry_config import INDUSTRIES

# ================================================
# CREATE API APP
# ================================================
app = FastAPI(
    title       = "ChurnGuard Pro API",
    description = (
        "AI-powered customer churn prediction API. "
        "Supports Telecom, Banking, and E-Commerce industries."
    ),
    version     = "1.0.0"
)

# Allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ================================================
# DATA MODELS — what the API accepts
# ================================================
class TelecomCustomer(BaseModel):
    tenure          : float = 12
    MonthlyCharges  : float = 65.0
    TotalCharges    : float = 780.0
    gender          : str   = "Male"
    SeniorCitizen   : int   = 0
    Partner         : str   = "Yes"
    Dependents      : str   = "No"
    PhoneService    : str   = "Yes"
    MultipleLines   : str   = "No"
    InternetService : str   = "DSL"
    Contract        : str   = "Month-to-month"
    PaperlessBilling: str   = "Yes"
    PaymentMethod   : str   = "Electronic check"

class BankCustomer(BaseModel):
    CreditScore    : float = 650
    Geography      : str   = "France"
    Gender         : str   = "Male"
    Age            : float = 35
    Tenure         : float = 3
    Balance        : float = 50000.0
    NumOfProducts  : int   = 1
    HasCrCard      : int   = 1
    IsActiveMember : int   = 1
    EstimatedSalary: float = 60000.0

class EcommerceCustomer(BaseModel):
    Tenure                      : float = 6
    PreferredLoginDevice        : str   = "Mobile Phone"
    CityTier                    : int   = 1
    WarehouseToHome             : float = 20
    PreferredPaymentMode        : str   = "Debit Card"
    Gender                      : str   = "Male"
    HourSpendOnApp              : float = 3
    NumberOfDeviceRegistered    : int   = 3
    PreferedOrderCat            : str   = "Mobile Phone"
    SatisfactionScore           : int   = 3
    MaritalStatus               : str   = "Single"
    NumberOfAddress             : int   = 3
    Complain                    : int   = 0
    OrderAmountHikeFromlastYear : float = 15
    CouponUsed                  : float = 2
    OrderCount                  : float = 3
    DaySinceLastOrder           : float = 5
    CashbackAmount              : float = 150

class PredictionResponse(BaseModel):
    industry         : str
    churn_probability: float
    prediction       : str
    risk_level       : str
    confidence       : str
    recommendations  : list

# ================================================
# HELPER FUNCTIONS
# ================================================
def get_risk_label(prob: float) -> str:
    if prob > 0.65:
        return "High Risk"
    elif prob > 0.40:
        return "Medium Risk"
    else:
        return "Low Risk"

def get_confidence(prob: float) -> str:
    if prob > 0.85 or prob < 0.15:
        return "Very High"
    elif prob > 0.70 or prob < 0.30:
        return "High"
    elif prob > 0.55 or prob < 0.45:
        return "Medium"
    else:
        return "Low"

def load_model(industry: str):
    """Load model, scaler, features for an industry."""
    if industry not in INDUSTRIES:
        raise HTTPException(
            status_code=400,
            detail=f"Industry must be one of: {list(INDUSTRIES.keys())}"
        )
    config = INDUSTRIES[industry]
    try:
        model  = joblib.load(config['model_file'])
        scaler = joblib.load(config['scaler_file'])
        feats  = joblib.load(config['features_file'])
        return model, scaler, feats
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model not found for {industry}. Run multi_train.py first. Error: {str(e)}"
        )

def build_telecom_features(customer: TelecomCustomer,
                            feature_names: list) -> pd.DataFrame:
    raw = {
        'gender'          : 1.0 if customer.gender == "Male" else 0.0,
        'SeniorCitizen'   : float(customer.SeniorCitizen),
        'Partner'         : 1.0 if customer.Partner == "Yes" else 0.0,
        'Dependents'      : 1.0 if customer.Dependents == "Yes" else 0.0,
        'tenure'          : float(customer.tenure),
        'PhoneService'    : 1.0 if customer.PhoneService == "Yes" else 0.0,
        'MultipleLines'   : float({'No phone service':0,'No':1,'Yes':2}.get(customer.MultipleLines,0)),
        'PaperlessBilling': 1.0 if customer.PaperlessBilling == "Yes" else 0.0,
        'MonthlyCharges'  : float(customer.MonthlyCharges),
        'TotalCharges'    : float(customer.TotalCharges),
        'InternetService_Fiber optic': 1.0 if customer.InternetService == "Fiber optic" else 0.0,
        'InternetService_No'         : 1.0 if customer.InternetService == "No" else 0.0,
        'Contract_One year'          : 1.0 if customer.Contract == "One year" else 0.0,
        'Contract_Two year'          : 1.0 if customer.Contract == "Two year" else 0.0,
        'PaymentMethod_Electronic check'        : 1.0 if customer.PaymentMethod == "Electronic check" else 0.0,
        'PaymentMethod_Mailed check'            : 1.0 if customer.PaymentMethod == "Mailed check" else 0.0,
        'PaymentMethod_Credit card (automatic)' : 1.0 if customer.PaymentMethod == "Credit card (automatic)" else 0.0,
        'OnlineSecurity_No internet service'    : 1.0 if customer.InternetService == "No" else 0.0,
        'OnlineSecurity_Yes'                    : 0.0,
        'OnlineBackup_No internet service'      : 1.0 if customer.InternetService == "No" else 0.0,
        'OnlineBackup_Yes'                      : 0.0,
        'DeviceProtection_No internet service'  : 1.0 if customer.InternetService == "No" else 0.0,
        'DeviceProtection_Yes'                  : 0.0,
        'TechSupport_No internet service'       : 1.0 if customer.InternetService == "No" else 0.0,
        'TechSupport_Yes'                       : 0.0,
        'StreamingTV_No internet service'       : 1.0 if customer.InternetService == "No" else 0.0,
        'StreamingTV_Yes'                       : 0.0,
        'StreamingMovies_No internet service'   : 1.0 if customer.InternetService == "No" else 0.0,
        'StreamingMovies_Yes'                   : 0.0,
        'total_services'                        : 0.0,
    }
    vals  = [raw.get(f, 0.0) for f in feature_names]
    return pd.DataFrame(
        [vals], columns=feature_names, dtype=float
    )

def build_banking_features(customer: BankCustomer,
                            feature_names: list) -> pd.DataFrame:
    raw = {
        'CreditScore'      : float(customer.CreditScore),
        'Age'              : float(customer.Age),
        'Tenure'           : float(customer.Tenure),
        'Balance'          : float(customer.Balance),
        'NumOfProducts'    : float(customer.NumOfProducts),
        'HasCrCard'        : float(customer.HasCrCard),
        'IsActiveMember'   : float(customer.IsActiveMember),
        'EstimatedSalary'  : float(customer.EstimatedSalary),
        'Geography_Germany': 1.0 if customer.Geography == "Germany" else 0.0,
        'Geography_Spain'  : 1.0 if customer.Geography == "Spain" else 0.0,
        'Gender_Male'      : 1.0 if customer.Gender == "Male" else 0.0,
    }
    vals = [raw.get(f, 0.0) for f in feature_names]
    return pd.DataFrame(
        [vals], columns=feature_names, dtype=float
    )

def build_ecommerce_features(customer: EcommerceCustomer,
                              feature_names: list) -> pd.DataFrame:
    raw = {
        'Tenure'                      : float(customer.Tenure),
        'CityTier'                    : float(customer.CityTier),
        'WarehouseToHome'             : float(customer.WarehouseToHome),
        'HourSpendOnApp'              : float(customer.HourSpendOnApp),
        'NumberOfDeviceRegistered'    : float(customer.NumberOfDeviceRegistered),
        'SatisfactionScore'           : float(customer.SatisfactionScore),
        'NumberOfAddress'             : float(customer.NumberOfAddress),
        'Complain'                    : float(customer.Complain),
        'OrderAmountHikeFromlastYear' : float(customer.OrderAmountHikeFromlastYear),
        'CouponUsed'                  : float(customer.CouponUsed),
        'OrderCount'                  : float(customer.OrderCount),
        'DaySinceLastOrder'           : float(customer.DaySinceLastOrder),
        'CashbackAmount'              : float(customer.CashbackAmount),
        'PreferredLoginDevice_Mobile Phone': 1.0 if customer.PreferredLoginDevice == "Mobile Phone" else 0.0,
        'PreferredLoginDevice_Phone'       : 1.0 if customer.PreferredLoginDevice == "Phone" else 0.0,
        'PreferredPaymentMode_Credit Card' : 1.0 if customer.PreferredPaymentMode == "Credit Card" else 0.0,
        'PreferredPaymentMode_Debit Card'  : 1.0 if customer.PreferredPaymentMode == "Debit Card" else 0.0,
        'PreferredPaymentMode_E wallet'    : 1.0 if customer.PreferredPaymentMode == "E wallet" else 0.0,
        'PreferredPaymentMode_UPI'         : 1.0 if customer.PreferredPaymentMode == "UPI" else 0.0,
        'PreferredPaymentMode_Cash on Delivery': 1.0 if customer.PreferredPaymentMode == "Cash on Delivery" else 0.0,
        'Gender_Male'                      : 1.0 if customer.Gender == "Male" else 0.0,
        'PreferedOrderCat_Grocery'         : 1.0 if customer.PreferedOrderCat == "Grocery" else 0.0,
        'PreferedOrderCat_Laptop & Accessory': 1.0 if customer.PreferedOrderCat == "Laptop & Accessory" else 0.0,
        'PreferedOrderCat_Mobile'          : 1.0 if customer.PreferedOrderCat == "Mobile" else 0.0,
        'PreferedOrderCat_Mobile Phone'    : 1.0 if customer.PreferedOrderCat == "Mobile Phone" else 0.0,
        'PreferedOrderCat_Others'          : 1.0 if customer.PreferedOrderCat == "Others" else 0.0,
        'MaritalStatus_Married'            : 1.0 if customer.MaritalStatus == "Married" else 0.0,
        'MaritalStatus_Single'             : 1.0 if customer.MaritalStatus == "Single" else 0.0,
    }
    vals = [raw.get(f, 0.0) for f in feature_names]
    return pd.DataFrame(
        [vals], columns=feature_names, dtype=float
    )

def make_recommendation(prob: float, pred: int,
                         industry: str) -> list:
    recs = []
    if pred == 1:
        if industry == 'telecom':
            recs.append("Offer contract upgrade with 20% discount")
            if prob > 0.7:
                recs.append("Assign dedicated retention agent immediately")
                recs.append("Schedule call within 24 hours")
        elif industry == 'banking':
            recs.append("Schedule personal relationship manager call")
            recs.append("Offer premium account upgrade")
        elif industry == 'ecommerce':
            recs.append("Send personalized discount voucher")
            recs.append("Increase cashback for next 3 orders")
        recs.append(f"Customer has {prob:.0%} churn probability — act now")
    else:
        recs.append("Customer is low risk — maintain service quality")
        recs.append("Consider upsell opportunity")
    return recs

# ================================================
# API ENDPOINTS
# ================================================

@app.get("/")
def root():
    """API home — shows available endpoints."""
    return {
        "name"       : "ChurnGuard Pro API",
        "version"    : "1.0.0",
        "description": "AI-powered churn prediction API",
        "industries" : ["telecom", "banking", "ecommerce"],
        "endpoints"  : {
            "GET  /"                          : "API info",
            "GET  /health"                    : "Health check",
            "GET  /industries"                : "List industries",
            "POST /predict/telecom"           : "Predict telecom churn",
            "POST /predict/banking"           : "Predict banking churn",
            "POST /predict/ecommerce"         : "Predict ecommerce churn",
            "POST /predict/bulk/{industry}"   : "Bulk predict from CSV",
            "GET  /model/info/{industry}"     : "Model information",
        }
    }

@app.get("/health")
def health_check():
    """Check if API and models are running."""
    status = {}
    for key in INDUSTRIES:
        try:
            joblib.load(INDUSTRIES[key]['model_file'])
            status[key] = "ready"
        except Exception:
            status[key] = "not trained"
    return {
        "status" : "running",
        "models" : status
    }

@app.get("/industries")
def list_industries():
    """List all supported industries."""
    return {
        key: {
            "name"       : config['name'],
            "icon"       : config['icon'],
            "description": config['description']
        }
        for key, config in INDUSTRIES.items()
    }

@app.get("/model/info/{industry}")
def model_info(industry: str):
    """Get model information for an industry."""
    if industry not in INDUSTRIES:
        raise HTTPException(status_code=404,
                            detail="Industry not found")
    config = INDUSTRIES[industry]
    try:
        feats  = joblib.load(config['features_file'])
        try:
            params = joblib.load(
                config['model_file'].replace('.pkl','_params.pkl')
            )
        except Exception:
            params = "Run multi_train.py to see parameters"
        return {
            "industry"      : industry,
            "name"          : config['name'],
            "features_count": len(feats),
            "features"      : feats,
            "best_params"   : params,
            "algorithm"     : "XGBoost with GridSearchCV",
            "evaluation"    : "5-fold cross-validation"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/telecom",
          response_model=PredictionResponse)
def predict_telecom(customer: TelecomCustomer):
    """Predict churn for a single telecom customer."""
    model, scaler, feats = load_model('telecom')
    row    = build_telecom_features(customer, feats)
    scaled = scaler.transform(row)
    prob   = float(model.predict_proba(scaled)[0][1])
    pred   = int(model.predict(scaled)[0])
    return PredictionResponse(
        industry          = "Telecom",
        churn_probability = round(prob, 4),
        prediction        = "WILL CHURN" if pred == 1 else "WILL STAY",
        risk_level        = get_risk_label(prob),
        confidence        = get_confidence(prob),
        recommendations   = make_recommendation(prob, pred, 'telecom')
    )

@app.post("/predict/banking",
          response_model=PredictionResponse)
def predict_banking(customer: BankCustomer):
    """Predict churn for a single banking customer."""
    model, scaler, feats = load_model('banking')
    row    = build_banking_features(customer, feats)
    scaled = scaler.transform(row)
    prob   = float(model.predict_proba(scaled)[0][1])
    pred   = int(model.predict(scaled)[0])
    return PredictionResponse(
        industry          = "Banking",
        churn_probability = round(prob, 4),
        prediction        = "WILL CHURN" if pred == 1 else "WILL STAY",
        risk_level        = get_risk_label(prob),
        confidence        = get_confidence(prob),
        recommendations   = make_recommendation(prob, pred, 'banking')
    )

@app.post("/predict/ecommerce",
          response_model=PredictionResponse)
def predict_ecommerce(customer: EcommerceCustomer):
    """Predict churn for a single ecommerce customer."""
    model, scaler, feats = load_model('ecommerce')
    row    = build_ecommerce_features(customer, feats)
    scaled = scaler.transform(row)
    prob   = float(model.predict_proba(scaled)[0][1])
    pred   = int(model.predict(scaled)[0])
    return PredictionResponse(
        industry          = "E-Commerce",
        churn_probability = round(prob, 4),
        prediction        = "WILL CHURN" if pred == 1 else "WILL STAY",
        risk_level        = get_risk_label(prob),
        confidence        = get_confidence(prob),
        recommendations   = make_recommendation(prob, pred, 'ecommerce')
    )

@app.post("/predict/bulk/{industry}")
async def predict_bulk(industry: str,
                        file: UploadFile = File(...)):
    """
    Predict churn for multiple customers from a CSV file.
    Upload a CSV with customer data columns for the selected industry.
    Returns predictions for every row.
    """
    if industry not in INDUSTRIES:
        raise HTTPException(
            status_code=400,
            detail=f"Industry must be one of: {list(INDUSTRIES.keys())}"
        )

    # Read uploaded CSV
    try:
        contents = await file.read()
        df       = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read CSV: {str(e)}"
        )

    model, scaler, feats = load_model(industry)

    # Build feature matrix
    results = []
    for idx, row in df.iterrows():
        try:
            raw  = {f: float(row[f]) if f in row.index else 0.0
                    for f in feats}
            vals = [raw.get(f, 0.0) for f in feats]
            X    = pd.DataFrame(
                [vals], columns=feats, dtype=float
            )
            scaled = scaler.transform(X)
            prob   = float(model.predict_proba(scaled)[0][1])
            pred   = int(model.predict(scaled)[0])

            results.append({
                "row"              : int(idx),
                "churn_probability": round(prob, 4),
                "prediction"       : "WILL CHURN" if pred==1 else "WILL STAY",
                "risk_level"       : get_risk_label(prob)
            })
        except Exception as row_err:
            results.append({
                "row"  : int(idx),
                "error": str(row_err)
            })

    churners = sum(
        1 for r in results if r.get('prediction') == 'WILL CHURN'
    )

    return {
        "industry"           : industry,
        "total_customers"    : len(results),
        "predicted_churners" : churners,
        "churn_rate"         : f"{churners/len(results)*100:.1f}%",
        "predictions"        : results
    }