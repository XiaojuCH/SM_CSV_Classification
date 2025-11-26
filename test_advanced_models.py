import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from lightgbm import LGBMClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
import warnings
warnings.filterwarnings('ignore')

# 读取数据
data_files = ['DH.csv', 'KD.csv', 'PS10.csv', 'PS10-H.csv', 'QZ.csv', 'YM.csv']
labels = ['DH', 'KD', 'PS10', 'PS10-H', 'QZ', 'YM']

all_data = []
all_labels = []

for file, label in zip(data_files, labels):
    df = pd.read_csv(file, header=None)
    all_data.append(df.values)
    all_labels.extend([label] * len(df))

X = np.vstack(all_data)
y = np.array(all_labels)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("="*80)
print("测试高级模型")
print("="*80)
print(f"当前LightGBM最佳: 81.65%\n")

# 测试模型
models = {
    'LightGBM (基线)': LGBMClassifier(
        n_estimators=1000,
        max_depth=15,
        learning_rate=0.1,
        num_leaves=80,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    ),

    'HistGradientBoosting': HistGradientBoostingClassifier(
        max_iter=1000,
        max_depth=15,
        learning_rate=0.1,
        max_leaf_nodes=80,
        random_state=42
    ),

    'ExtraTrees': ExtraTreesClassifier(
        n_estimators=1000,
        max_depth=20,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    ),

    'LightGBM (激进)': LGBMClassifier(
        n_estimators=2000,
        max_depth=25,
        learning_rate=0.05,
        num_leaves=150,
        min_child_samples=10,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=0.05,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
}

results = {}

for name, model in models.items():
    print(f"\n{'='*80}")
    print(f"训练: {name}")
    print(f"{'='*80}")

    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)

    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, n_jobs=-1)

    results[name] = {
        'test_acc': test_acc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }

    print(f"测试准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"交叉验证: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    improvement = (test_acc - 0.8165) * 100
    if improvement > 0:
        print(f"✅ 提升: +{improvement:.2f}%")
    else:
        print(f"❌ 下降: {improvement:.2f}%")

print("\n" + "="*80)
print("结果排名")
print("="*80)

sorted_results = sorted(results.items(), key=lambda x: x[1]['test_acc'], reverse=True)

for i, (name, result) in enumerate(sorted_results, 1):
    print(f"{i}. {name}: {result['test_acc']*100:.2f}% (CV: {result['cv_mean']*100:.2f}%)")

best_name = sorted_results[0][0]
best_acc = sorted_results[0][1]['test_acc']

print(f"\n🏆 最佳模型: {best_name}")
print(f"准确率: {best_acc*100:.2f}%")
