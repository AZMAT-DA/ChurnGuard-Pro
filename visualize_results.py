# visualize_results.py
# Run this after train_model.py to generate comparison charts

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import joblib
import numpy as np

print("Generating charts...")

# =============================================
# CHART 1: Model Comparison Bar Chart
# =============================================
try:
    comparison_df = pd.read_csv('model_comparison.csv')
    print(f"Loaded {len(comparison_df)} models from model_comparison.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Phase 1 — Model Comparison', fontsize=16, fontweight='bold')

    # Left chart: all metrics side by side
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    x       = np.arange(len(comparison_df))
    width   = 0.18
    colors  = ['#3B82F6', '#8B5CF6', '#10B981', '#F59E0B']

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        axes[0].bar(x + i * width, comparison_df[metric],
                    width, label=metric, color=color, alpha=0.85)

    axes[0].set_xlabel('Model', fontsize=12)
    axes[0].set_ylabel('Score (%)', fontsize=12)
    axes[0].set_title('All Metrics Comparison', fontsize=13)
    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(comparison_df['Model'], rotation=15, ha='right')
    axes[0].legend(fontsize=10)
    axes[0].set_ylim(0, 100)
    axes[0].grid(axis='y', alpha=0.3)

    # Right chart: AUC score only with value labels
    bar_colors = ['#60A5FA', '#A78BFA', '#34D399', '#FBBF24'][:len(comparison_df)]
    bars = axes[1].bar(comparison_df['Model'], comparison_df['AUC'],
                       color=bar_colors, alpha=0.85, width=0.5)

    for bar in bars:
        h = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, h + 0.3,
                     f'{h:.1f}%', ha='center', va='bottom',
                     fontweight='bold', fontsize=11)

    axes[1].set_title('AUC Score by Model', fontsize=13)
    axes[1].set_ylabel('AUC Score (%)', fontsize=12)
    axes[1].set_ylim(0, 100)
    axes[1].set_xticklabels(comparison_df['Model'], rotation=15, ha='right')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_comparison_chart.png', dpi=150, bbox_inches='tight')
    print("Saved: model_comparison_chart.png")
    plt.close()

except Exception as e:
    print(f"Chart 1 error: {e}")

# =============================================
# CHART 2: Neural Network Training Loss
# =============================================
try:
    train_losses = joblib.load('nn_history.pkl')

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, color='#3B82F6', linewidth=2, label='Training Loss')
    plt.title('Neural Network — Training Loss over 50 Epochs',
              fontsize=13, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('neural_network_training.png', dpi=150, bbox_inches='tight')
    print("Saved: neural_network_training.png")
    plt.close()

except Exception as e:
    print(f"Chart 2 skipped (Neural Network not trained yet): {e}")

# =============================================
# CHART 3: Best Model Feature Importance
# =============================================
try:
    import joblib
    from xgboost import XGBClassifier

    model        = joblib.load('churn_model.pkl')
    feature_names = joblib.load('feature_names.pkl')

    importances = model.feature_importances_
    feat_df     = pd.DataFrame({
        'Feature'   : feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=True).tail(15)

    plt.figure(figsize=(10, 7))
    bars = plt.barh(feat_df['Feature'], feat_df['Importance'],
                    color='#8B5CF6', alpha=0.85)
    plt.title('Top 15 Most Important Features — XGBoost Tuned',
              fontsize=13, fontweight='bold')
    plt.xlabel('Importance Score', fontsize=12)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
    print("Saved: feature_importance.png")
    plt.close()

except Exception as e:
    print(f"Chart 3 error: {e}")

# =============================================
# FINAL SUMMARY PRINTED IN TERMINAL
# =============================================
print("\n" + "=" * 50)
print("PHASE 1 RESULTS SUMMARY")
print("=" * 50)

try:
    df = pd.read_csv('model_comparison.csv')
    print(df.to_string(index=False))

    best = df.loc[df['AUC'].idxmax()]
    print(f"\nBest model  : {best['Model']}")
    print(f"Best AUC    : {best['AUC']}%")
    print(f"Best Recall : {best['Recall']}%")
    print(f"Best F1     : {best['F1-Score']}%")
except Exception as e:
    print(f"Summary error: {e}")

print("\nAll done! Open these files to see your charts:")
print("  model_comparison_chart.png")
print("  neural_network_training.png")
print("  feature_importance.png")