# shap_explainer.py — Works with any industry model

import shap
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import io


def get_shap_explanation(model, input_row: pd.DataFrame) -> dict:
    """
    Calculate SHAP values for any XGBoost model.
    Now accepts model as parameter instead of loading fixed model.
    """
    explainer  = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_row)
    sv          = shap_values[0]
    feature_names = input_row.columns.tolist()
    shap_series = pd.Series(sv, index=feature_names)
    shap_sorted = shap_series.reindex(
        shap_series.abs().sort_values(ascending=False).index
    )
    risk_increasers = shap_sorted[shap_sorted > 0].head(5)
    risk_decreasers = shap_sorted[shap_sorted < 0].head(5)

    return {
        'shap_values'    : shap_series,
        'risk_increasers': risk_increasers,
        'risk_decreasers': risk_decreasers,
        'top_features'   : shap_sorted.head(8)
    }


def plot_shap_bar(shap_explanation: dict) -> bytes:
    """Create SHAP bar chart and return as PNG bytes."""
    top      = shap_explanation['top_features']
    names    = [n.replace('_', ' ')[:30] for n in top.index]
    vals     = top.values
    colors   = ['#E24B4A' if v > 0 else '#1A7A4A' for v in vals]

    fig, ax  = plt.subplots(figsize=(8, 5))
    ax.barh(range(len(names)), vals,
            color=colors, alpha=0.85, height=0.6)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.axvline(x=0, color='gray', linewidth=0.8)
    ax.set_xlabel('SHAP value (impact on churn prediction)',
                  fontsize=10)
    ax.set_title('Why did the model predict this?',
                 fontsize=12, fontweight='bold', pad=12)

    for i, (bar_val) in enumerate(vals):
        label = f"+{bar_val:.3f}" if bar_val > 0 else f"{bar_val:.3f}"
        x_pos = bar_val + 0.001 if bar_val > 0 else bar_val - 0.001
        ha    = 'left' if bar_val > 0 else 'right'
        ax.text(x_pos, i, label, va='center',
                ha=ha, fontsize=9)

    legend = [
        Patch(color='#E24B4A', alpha=0.85,
              label='Increases churn risk'),
        Patch(color='#1A7A4A', alpha=0.85,
              label='Decreases churn risk')
    ]
    ax.legend(handles=legend, loc='lower right', fontsize=9)
    ax.invert_yaxis()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    buf.seek(0)
    plt.close()
    return buf.getvalue()


def get_shap_recommendations(shap_explanation: dict,
                              pred: int,
                              industry: str = 'telecom') -> list:
    """Generate recommendations based on SHAP values for any industry."""
    recommendations = []
    risk_increasers = shap_explanation['risk_increasers']
    risk_decreasers = shap_explanation['risk_decreasers']

    if pred == 1:
        for feature, value in risk_increasers.items():
            impact = round(abs(value) * 100, 1)

            # Telecom patterns
            if 'Contract' in feature:
                recommendations.append(
                    f"Contract type is the biggest churn driver "
                    f"(+{impact}% risk). Offer upgrade with 20% discount."
                )
            elif 'tenure' in feature.lower():
                recommendations.append(
                    f"Short tenure increases churn risk (+{impact}%). "
                    "Assign dedicated onboarding support agent."
                )
            elif 'MonthlyCharges' in feature:
                recommendations.append(
                    f"High monthly charges driving churn (+{impact}%). "
                    "Offer 10-15% bill reduction bundle."
                )
            elif 'Fiber' in feature:
                recommendations.append(
                    f"Fiber optic service is a risk factor (+{impact}%). "
                    "Check service quality proactively."
                )
            elif 'Electronic check' in feature:
                recommendations.append(
                    f"Electronic check payment increases risk (+{impact}%). "
                    "Suggest switching to automatic payment."
                )

            # Banking patterns
            elif 'NumOfProducts' in feature or 'Products' in feature:
                recommendations.append(
                    f"Number of products is a churn driver (+{impact}%). "
                    "Review product fit and simplify customer portfolio."
                )
            elif 'Age' in feature:
                recommendations.append(
                    f"Customer age is a risk factor (+{impact}%). "
                    "Assign dedicated relationship manager."
                )
            elif 'IsActiveMember' in feature or 'Active' in feature:
                recommendations.append(
                    f"Low activity is increasing churn risk (+{impact}%). "
                    "Send re-engagement campaign with exclusive offer."
                )
            elif 'Balance' in feature:
                recommendations.append(
                    f"Account balance pattern increases risk (+{impact}%). "
                    "Offer incentive to maintain healthy balance."
                )
            elif 'Geography' in feature:
                recommendations.append(
                    f"Geographic region is a churn factor (+{impact}%). "
                    "Offer region-specific loyalty program."
                )

            # E-Commerce patterns
            elif 'Complain' in feature:
                recommendations.append(
                    f"Customer complaint is a top churn driver (+{impact}%). "
                    "Resolve immediately and offer compensation coupon."
                )
            elif 'Satisfaction' in feature:
                recommendations.append(
                    f"Low satisfaction score drives churn (+{impact}%). "
                    "Send personal apology and discount voucher."
                )
            elif 'DaySinceLastOrder' in feature:
                recommendations.append(
                    f"Days since last order increases risk (+{impact}%). "
                    "Send personalized product recommendations now."
                )
            elif 'CashbackAmount' in feature:
                recommendations.append(
                    f"Low cashback is a risk factor (+{impact}%). "
                    "Increase cashback offer to re-engage customer."
                )
            elif 'HourSpendOnApp' in feature or 'Hour' in feature:
                recommendations.append(
                    f"Low app engagement increases risk (+{impact}%). "
                    "Send push notification with personalized offers."
                )
            elif 'WarehouseToHome' in feature:
                recommendations.append(
                    f"Long delivery distance is a risk factor (+{impact}%). "
                    "Offer free shipping or faster delivery option."
                )

        if not recommendations:
            recommendations.append(
                "Multiple factors contributing to churn risk. "
                "Schedule personal retention call within 72 hours."
            )
        recommendations.append(
            "Contact customer within 48 hours with a personalized "
            "retention offer addressing the above risk factors."
        )

    else:
        recommendations.append(
            "Customer shows low churn risk. "
            "Continue current service quality."
        )
        for feature, value in risk_decreasers.items():
            impact = round(abs(value) * 100, 1)
            if 'Contract' in feature or 'tenure' in feature.lower():
                recommendations.append(
                    f"Long-term loyalty is keeping this customer "
                    f"(-{impact}% risk). Reward with loyalty benefit."
                )
                break
            elif 'IsActiveMember' in feature or 'Active' in feature:
                recommendations.append(
                    f"High activity reduces churn risk (-{impact}%). "
                    "Consider VIP program enrollment."
                )
                break
            elif 'Satisfaction' in feature:
                recommendations.append(
                    f"High satisfaction reduces risk (-{impact}%). "
                    "Ask for review and referral."
                )
                break

    return recommendations[:5]