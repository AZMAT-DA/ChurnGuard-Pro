# train_model.py - PHASE 1 UPGRADED VERSION (PyTorch)

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, roc_auc_score,
                             precision_score, recall_score,
                             f1_score, accuracy_score)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# =============================================
# STEP 1: LOAD DATA
# =============================================
print("=" * 50)
print("STEP 1: LOADING DATA")
print("=" * 50)

df = pd.read_csv('cleaned_data.csv')
df = df.dropna()

X = df.drop('Churn', axis=1)
y = df['Churn']

print(f"Total customers : {len(df)}")
print(f"Total features  : {X.shape[1]}")
print(f"Churners        : {y.sum()}")
print(f"Non-churners    : {(y == 0).sum()}")

# =============================================
# STEP 2: SPLIT DATA
# =============================================
print("\n" + "=" * 50)
print("STEP 2: SPLITTING DATA (80% train / 20% test)")
print("=" * 50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(f"Training rows : {len(X_train)}")
print(f"Testing rows  : {len(X_test)}")

# =============================================
# STEP 3: SMOTE BALANCING
# =============================================
print("\n" + "=" * 50)
print("STEP 3: BALANCING WITH SMOTE")
print("=" * 50)

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(f"Stayed  (0): {(y_train_bal == 0).sum()}")
print(f"Churned (1): {(y_train_bal == 1).sum()}")

# We will store all results here for final comparison
results = {}

# =============================================
# MODEL 1: LOGISTIC REGRESSION
# =============================================
print("\n" + "=" * 50)
print("MODEL 1: Logistic Regression (simple baseline)")
print("=" * 50)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_bal, y_train_bal)

lr_pred = lr.predict(X_test)
lr_prob = lr.predict_proba(X_test)[:, 1]
lr_auc  = roc_auc_score(y_test, lr_prob)

print(classification_report(y_test, lr_pred, target_names=['Stay', 'Churn']))
print(f"AUC Score: {lr_auc:.4f}")

results['Logistic Regression'] = {
    'pred': lr_pred,
    'prob': lr_prob,
    'auc' : lr_auc
}

# =============================================
# MODEL 2: RANDOM FOREST (improved settings)
# =============================================
print("\n" + "=" * 50)
print("MODEL 2: Random Forest (improved)")
print("=" * 50)

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    random_state=42
)
rf.fit(X_train_bal, y_train_bal)

rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]
rf_auc  = roc_auc_score(y_test, rf_prob)

print(classification_report(y_test, rf_pred, target_names=['Stay', 'Churn']))
print(f"AUC Score: {rf_auc:.4f}")

results['Random Forest'] = {
    'pred': rf_pred,
    'prob': rf_prob,
    'auc' : rf_auc
}

# =============================================
# MODEL 3: XGBOOST TUNED
# =============================================
print("\n" + "=" * 50)
print("MODEL 3: XGBoost (tuned - better than before)")
print("=" * 50)
print("Changes from old version:")
print("  n_estimators : 100  -> 300  (more trees)")
print("  learning_rate: 0.3  -> 0.05 (more careful learning)")
print("  max_depth    : 6    -> 5    (less overfitting)")
print("  + subsample, colsample_bytree, gamma added")

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    eval_metric='logloss',
    random_state=42
)
xgb.fit(X_train_bal, y_train_bal)

xgb_pred = xgb.predict(X_test)
xgb_prob = xgb.predict_proba(X_test)[:, 1]
xgb_auc  = roc_auc_score(y_test, xgb_prob)

print(classification_report(y_test, xgb_pred, target_names=['Stay', 'Churn']))
print(f"AUC Score: {xgb_auc:.4f}")

results['XGBoost Tuned'] = {
    'pred': xgb_pred,
    'prob': xgb_prob,
    'auc' : xgb_auc
}

# =============================================
# MODEL 4: NEURAL NETWORK (PyTorch)
# =============================================
print("\n" + "=" * 50)
print("MODEL 4: Neural Network (Deep Learning with PyTorch)")
print("=" * 50)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(42)

    # Convert data to PyTorch tensors
    # A tensor is like a numpy array but for deep learning
    X_tr = torch.FloatTensor(X_train_bal.values)
    y_tr = torch.FloatTensor(y_train_bal.values)
    X_te = torch.FloatTensor(X_test.values)

    # Create DataLoader — feeds data in batches of 32
    dataset = TensorDataset(X_tr, y_tr)
    loader  = DataLoader(dataset, batch_size=32, shuffle=True)

    # Define Neural Network structure
    class ChurnNet(nn.Module):
        def __init__(self, input_size):
            super(ChurnNet, self).__init__()

            # Each Dense layer is called Linear in PyTorch
            self.network = nn.Sequential(
                # Layer 1: input -> 64 neurons
                nn.Linear(input_size, 64),
                nn.ReLU(),           # activation function
                nn.Dropout(0.3),     # randomly drop 30% to prevent overfitting

                # Layer 2: 64 -> 32 neurons
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.3),

                # Layer 3: 32 -> 16 neurons
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Dropout(0.2),

                # Output: 16 -> 1 neuron (probability)
                nn.Linear(16, 1),
                nn.Sigmoid()         # converts output to 0-1 probability
            )

        def forward(self, x):
            return self.network(x)

    # Create the model
    input_size = X_tr.shape[1]
    nn_model   = ChurnNet(input_size)

    print(f"Neural Network created with {input_size} input features")
    print(f"Architecture: {input_size} -> 64 -> 32 -> 16 -> 1")
    print(f"Total parameters: {sum(p.numel() for p in nn_model.parameters())}")

    # Loss function and optimizer
    criterion = nn.BCELoss()                          # Binary Cross Entropy
    optimizer = torch.optim.Adam(nn_model.parameters(), lr=0.001)

    # Training loop
    print("\nTraining Neural Network (50 epochs)...")
    train_losses = []

    for epoch in range(50):
        nn_model.train()
        epoch_loss = 0

        for batch_X, batch_y in loader:
            optimizer.zero_grad()                     # clear old gradients
            output = nn_model(batch_X).squeeze()      # forward pass
            loss   = criterion(output, batch_y)       # calculate error
            loss.backward()                           # backpropagation
            optimizer.step()                          # update weights
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        train_losses.append(avg_loss)

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:2d}/50 — Loss: {avg_loss:.4f}")

    print("Training complete!")

    # Predict on test set
    nn_model.eval()
    with torch.no_grad():
        nn_prob = nn_model(X_te).squeeze().numpy()

    nn_pred = (nn_prob > 0.5).astype(int)
    nn_auc  = roc_auc_score(y_test, nn_prob)

    print("\nNeural Network Results:")
    print(classification_report(y_test, nn_pred, target_names=['Stay', 'Churn']))
    print(f"AUC Score: {nn_auc:.4f}")

    results['Neural Network'] = {
        'pred': nn_pred,
        'prob': nn_prob,
        'auc' : nn_auc
    }

    # Save neural network and training history
    torch.save(nn_model.state_dict(), 'neural_network_model.pth')
    joblib.dump(train_losses, 'nn_history.pkl')
    joblib.dump(input_size,   'nn_input_size.pkl')
    print("\nNeural Network saved as neural_network_model.pth")

except ImportError:
    print("PyTorch not installed. Run: pip install torch")
except Exception as e:
    print(f"Neural Network error: {e}")
    print("Continuing without Neural Network...")

# =============================================
# FINAL COMPARISON TABLE
# =============================================
print("\n" + "=" * 50)
print("FINAL COMPARISON — ALL MODELS")
print("=" * 50)

rows = []
for name, res in results.items():
    p = res['pred']
    rows.append({
        'Model'    : name,
        'Accuracy' : round(accuracy_score(y_test, p)  * 100, 1),
        'Precision': round(precision_score(y_test, p) * 100, 1),
        'Recall'   : round(recall_score(y_test, p)    * 100, 1),
        'F1-Score' : round(f1_score(y_test, p)        * 100, 1),
        'AUC'      : round(res['auc']                 * 100, 1)
    })

comparison_df = pd.DataFrame(rows)
print(comparison_df.to_string(index=False))

# Save to CSV
comparison_df.to_csv('model_comparison.csv', index=False)
print("\nComparison table saved to model_comparison.csv")

# =============================================
# SAVE BEST MODEL
# =============================================
best_row  = comparison_df.loc[comparison_df['AUC'].idxmax()]
best_name = best_row['Model']
best_auc  = best_row['AUC']

print(f"\nBest model by AUC: {best_name} ({best_auc}%)")

# Always save XGBoost as main model — most reliable for tabular data
joblib.dump(xgb, 'churn_model.pkl')
joblib.dump(X_train.columns.tolist(), 'feature_names.pkl')

print("\n" + "=" * 50)
print("PHASE 1 COMPLETE!")
print("=" * 50)
print("Files saved:")
print("  churn_model.pkl             — tuned XGBoost (used by app)")
print("  feature_names.pkl           — column names")
print("  neural_network_model.pth    — neural network weights")
print("  nn_history.pkl              — training loss per epoch")
print("  model_comparison.csv        — all model scores")