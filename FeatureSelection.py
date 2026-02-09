import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import os

attack_types = [
    "Bot", "DDoS", "DoS GoldenEye", "DoS Hulk",
    "DoS Slowhttptest", "DoS slowloris",
    "Heartbleed", "Infiltration", "PortScan"
]

for attack in attack_types:
    file_name = f"{attack}_compressed.csv"

    if not os.path.exists(file_name):
        print(f"File not found: {file_name}")
        continue

    print(f"\nProcessing: {file_name}")
    df = pd.read_csv(file_name)

    #  Clean column names
    df.columns = df.columns.str.strip()

    # Binary labels
    X = df.drop(columns=["Label"])
    X = X.replace([np.inf, -np.inf], 0)
    X = X.fillna(0)

    y = df["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

    print("Label distribution:")
    print(y.value_counts())

    clf = RandomForestClassifier(
        n_estimators=30,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )

    clf.fit(X, y)

    importances = clf.feature_importances_

    imp_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    # Save CSV
    out_csv = f"{attack}_feature_importance.csv"
    imp_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # Plot Top 20
    top20 = imp_df.head(20)

    plt.figure(figsize=(12, 6))
    plt.bar(top20["Feature"], top20["Importance"])
    plt.title(f"Top 20 Important Features - {attack}")
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

print("\n Feature selection + graphs completed!")
