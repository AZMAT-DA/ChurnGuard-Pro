# recommendations.py — Smart retention recommendation engine

def get_recommendations(customer_data: dict, prob: float, pred: int) -> list:
    """
    Analyze customer data and return specific actionable recommendations.
    """
    recommendations = []

    contract  = str(customer_data.get('contract', ''))
    internet  = str(customer_data.get('internet', ''))
    tenure_raw = customer_data.get('tenure', '0 months')
    tenure    = int(str(tenure_raw).replace(' months', '').replace('months','').strip()) if tenure_raw else 0
    monthly_raw = customer_data.get('monthly_charges', '0')
    monthly   = float(str(monthly_raw).replace('$', '').strip()) if monthly_raw else 0
    senior    = str(customer_data.get('senior', 'No'))
    tech      = str(customer_data.get('tech_support', 'No'))
    paperless = str(customer_data.get('paperless', 'No'))

    if pred == 1:
        if 'Month-to-month' in contract:
            recommendations.append(
                "Offer a 20% discount to upgrade from Month-to-month to a "
                "One Year contract — month-to-month customers churn at 42% "
                "vs only 11% for annual contracts."
            )
        if tenure < 6:
            recommendations.append(
                "This is a new customer (under 6 months). Assign a dedicated "
                "onboarding support agent and check in within 48 hours to "
                "ensure they are satisfied with the service."
            )
        elif tenure < 12:
            recommendations.append(
                "Customer is in their first year — a critical retention period. "
                "Send a loyalty appreciation message and offer a free service "
                "upgrade for 3 months."
            )
        if monthly > 80:
            recommendations.append(
                f"Monthly charges are high (${monthly:.0f}/month). Offer a "
                "customized bundle that reduces the bill by 10-15% while "
                "maintaining core services."
            )
        if 'Fiber optic' in internet:
            recommendations.append(
                "Fiber optic customers have the highest churn rate (41%). "
                "Proactively check for service quality issues and offer a "
                "free speed upgrade or a month of free service."
            )
        if senior == 'Yes':
            recommendations.append(
                "Senior customer — offer a dedicated senior support helpline "
                "and simplified billing. Senior-specific plans with lower "
                "rates can significantly improve retention."
            )
        if tech == 'No':
            recommendations.append(
                "Customer does not have Tech Support. Offer a free 3-month "
                "trial of Tech Support — customers with this service churn "
                "significantly less."
            )
        if paperless == 'Yes':
            recommendations.append(
                "Customer uses paperless billing — ensure they are receiving "
                "and reading their bills. Send a clear bill summary email "
                "monthly to avoid billing surprise-related churn."
            )
        if not recommendations:
            recommendations.append(
                "Schedule a personal customer satisfaction call within 3 days. "
                "Ask about their experience and address any concerns directly."
            )
            recommendations.append(
                "Offer a loyalty reward — free month, service upgrade, or "
                "exclusive discount for staying with the company."
            )
    else:
        recommendations.append(
            "Customer shows low churn risk. Continue providing consistent "
            "service quality to maintain their satisfaction."
        )
        if tenure > 24:
            recommendations.append(
                "Long-term loyal customer. Consider enrolling them in a "
                "loyalty rewards program to further strengthen retention."
            )
        if monthly < 50:
            recommendations.append(
                "Low monthly spend — this customer may benefit from and "
                "appreciate an upsell to a premium bundle with added features."
            )
        recommendations.append(
            "Schedule a routine satisfaction check-in every 6 months "
            "to maintain the relationship proactively."
        )

    return recommendations


def get_risk_label(prob: float) -> str:
    """Returns a risk level label."""
    if prob > 0.65:
        return "High Risk"
    elif prob > 0.4:
        return "Medium Risk"
    else:
        return "Low Risk"


def get_risk_color(prob: float) -> str:
    """Returns a color string based on churn probability."""
    if prob > 0.65:
        return "red"
    elif prob > 0.4:
        return "orange"
    else:
        return "green"