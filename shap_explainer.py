# shap_explainer.py — SHAP-based model explanations

import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

# Load model and feature names once
model         = joblib.load('churn_model.pkl')
feature_names = joblib.load('feature_names.pkl')

# Create SHAP explainer — TreeExplainer works best with XGBoost
explainer = shap.TreeExplainer(model)


def get_shap_explanation(input_row: pd.DataFrame) -> dict:
    """
    Calculate SHAP values for a single customer prediction.
    Returns a dict with top reasons pushing toward and away from churn.
    """
    # Calculate SHAP values
    shap_values = explainer.shap_values(input_row)

    # shap_values is an array of shape (1, n_features)
    # Each value shows how much that feature pushed the prediction
    sv = shap_values[0]  # get first (only) row

    # Create a series mapping feature names to their SHAP values
    shap_series = pd.Series(sv, index=feature_names)

    # Sort by absolute value to find most important features
    shap_sorted = shap_series.reindex(
        shap_series.abs().sort_values(ascending=False).index
    )

    # Split into features that increase churn risk (positive)
    # and features that decrease churn risk (negative)
    risk_increasers = shap_sorted[shap_sorted > 0].head(5)
    risk_decreasers = shap_sorted[shap_sorted < 0].head(5)

    return {
        'shap_values'     : shap_series,
        'risk_increasers' : risk_increasers,
        'risk_decreasers' : risk_decreasers,
        'top_features'    : shap_sorted.head(8)
    }


def get_shap_recommendations(shap_explanation: dict,
                              pred: int) -> list:
    """
    Generate data-driven recommendations based on SHAP values.
    These come directly from what the model learned — not manual rules.
    """
    recommendations = []
    risk_increasers = shap_explanation['risk_increasers']

    if pred == 1:
        # Go through each factor increasing churn risk
        for feature, value in risk_increasers.items():
            impact_pct = round(abs(value) * 100, 1)

            # Contract type
            if 'Contract_One year' in feature or 'Contract_Two year' in feature:
                recommendations.append(
                    f"Contract type is the biggest churn driver "
                    f"(+{impact_pct}% risk). Offer an upgrade to a "
                    f"One Year or Two Year contract with a 15-20% discount."
                )

            elif 'tenure' in feature.lower():
                recommendations.append(
                    f"Short tenure is increasing churn risk "
                    f"(+{impact_pct}% risk). This customer needs immediate "
                    f"onboarding attention — assign a dedicated support agent."
                )

            elif 'MonthlyCharges' in feature or 'TotalCharges' in feature:
                recommendations.append(
                    f"High charges are pushing churn risk up "
                    f"(+{impact_pct}% risk). Offer a customized bundle "
                    f"that reduces monthly bill by 10-15%."
                )

            elif 'Fiber optic' in feature:
                recommendations.append(
                    f"Fiber optic service is a churn risk factor "
                    f"(+{impact_pct}% risk). Proactively check service "
                    f"quality and offer a free speed upgrade."
                )

            elif 'Electronic check' in feature:
                recommendations.append(
                    f"Electronic check payment increases churn risk "
                    f"(+{impact_pct}% risk). Encourage switching to "
                    f"automatic bank transfer or credit card payment."
                )

            elif 'TechSupport' in feature:
                recommendations.append(
                    f"Lack of tech support is increasing risk "
                    f"(+{impact_pct}% risk). Offer a free 3-month "
                    f"trial of Tech Support service."
                )

            elif 'SeniorCitizen' in feature:
                recommendations.append(
                    f"Senior citizen status increases churn risk "
                    f"(+{impact_pct}% risk). Offer a senior-specific "
                    f"plan with simplified billing and dedicated support."
                )

            elif 'PaperlessBilling' in feature:
                recommendations.append(
                    f"Paperless billing is a risk factor "
                    f"(+{impact_pct}% risk). Ensure monthly bill "
                    f"summary emails are being delivered and read."
                )

        # Add a general recommendation if we found risk factors
        if recommendations:
            recommendations.append(
                "Schedule a personal retention call within 72 hours "
                "to address the above risk factors directly."
            )

        # Fallback if no specific recommendations generated
        if not recommendations:
            recommendations.append(
                "Multiple factors are contributing to churn risk. "
                "Schedule a full customer review and offer a loyalty package."
            )

    else:
        # Low risk — explain what is keeping them loyal
        risk_decreasers = shap_explanation['risk_decreasers']

        recommendations.append(
            "This customer shows low churn risk. "
            "Continue current service quality."
        )

        for feature, value in risk_decreasers.items():
            impact_pct = round(abs(value) * 100, 1)

            if 'Contract' in feature:
                recommendations.append(
                    f"Long-term contract is keeping this customer loyal "
                    f"(-{impact_pct}% risk reduction). Reward them with "
                    f"a loyalty benefit at renewal time."
                )
                break
            elif 'tenure' in feature.lower():
                recommendations.append(
                    f"Long tenure is a strong loyalty indicator "
                    f"(-{impact_pct}% risk reduction). Consider enrolling "
                    f"them in a VIP rewards program."
                )
                break

    return recommendations[:5]  # return max 5 recommendations


def plot_shap_bar(shap_explanation: dict) -> bytes:
    """
    Create a horizontal bar chart showing SHAP feature impacts.
    Returns chart as PNG bytes for Streamlit display.
    """
    top_features = shap_explanation['top_features']

    # Clean up feature names for display
    display_names = []
    for name in top_features.index:
        clean = (name
                 .replace('_', ' ')
                 .replace('Contract One year', 'Contract: One Year')
                 .replace('Contract Two year', 'Contract: Two Year')
                 .replace('InternetService Fiber optic', 'Internet: Fiber Optic')
                 .replace('InternetService No', 'Internet: No Service')
                 .replace('PaymentMethod Electronic check', 'Payment: Electronic Check')
                 .replace('PaymentMethod Credit card (automatic)', 'Payment: Credit Card')
                 .replace('PaymentMethod Mailed check', 'Payment: Mailed Check')
                 .replace('MonthlyCharges', 'Monthly Charges')
                 .replace('TotalCharges', 'Total Charges')
                 .replace('SeniorCitizen', 'Senior Citizen')
                 .replace('PaperlessBilling', 'Paperless Billing')
                 .replace('PhoneService', 'Phone Service')
                 .replace('TechSupport Yes', 'Tech Support')
                 .replace('OnlineSecurity Yes', 'Online Security')
                 )
        display_names.append(clean[:35])  # truncate long names

    values = top_features.values
    colors = ['#E24B4A' if v > 0 else '#1A7A4A' for v in values]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(
        range(len(display_names)),
        values,
        color=colors,
        alpha=0.85,
        height=0.6
    )

    ax.set_yticks(range(len(display_names)))
    ax.set_yticklabels(display_names, fontsize=10)
    ax.axvline(x=0, color='gray', linewidth=0.8, linestyle='-')
    ax.set_xlabel('SHAP value (impact on churn prediction)', fontsize=10)
    ax.set_title(
        'Why did the model make this prediction?',
        fontsize=12, fontweight='bold', pad=12
    )

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, values)):
        label = f"+{val:.3f}" if val > 0 else f"{val:.3f}"
        x_pos = val + 0.001 if val > 0 else val - 0.001
        ha    = 'left' if val > 0 else 'right'
        ax.text(x_pos, i, label, va='center', ha=ha, fontsize=9)

    # Legend
    from matplotlib.patches import Patch
    legend = [
        Patch(color='#E24B4A', alpha=0.85, label='Increases churn risk'),
        Patch(color='#1A7A4A', alpha=0.85, label='Decreases churn risk')
    ]
    ax.legend(handles=legend, loc='lower right', fontsize=9)

    ax.invert_yaxis()
    plt.tight_layout()

    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf.getvalue()