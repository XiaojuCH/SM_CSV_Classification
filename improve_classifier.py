import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取所有数据文件
data_files = ['DH.csv', 'KD.csv', 'PS10.csv', 'PS10-H.csv', 'QZ.csv', 'YM.csv']
labels = ['DH', 'KD', 'PS10', 'PS10-H', 'QZ', 'YM']

# 加载数据
all_data = []
all_labels = []

for file, label in zip(data_files, labels):
    df = pd.read_csv(file, header=None)
    all_data.append(df.values)
    all_labels.extend([label] * len(df))

# 合并数据
X = np.vstack(all_data)
y = np.array(all_labels)

print("="*80)
print("特征工程优化")
print("="*80)
print(f"\n原始特征数: {X.shape[1]}")

# ============================================================================
# 特征工程：创建新特征
# ============================================================================

def create_enhanced_features(X):
    """创建增强特征"""
    X_new = X.copy()

    # 1. 特征统计量
    feature_mean = X.mean(axis=1, keepdims=True)
    feature_std = X.std(axis=1, keepdims=True)
    feature_max = X.max(axis=1, keepdims=True)
    feature_min = X.min(axis=1, keepdims=True)

    # 2. 特征比值（针对混淆类别）
    # 基于领域知识，创建一些可能有区分度的比值特征
    ratios = []
    for i in range(0, X.shape[1]-1, 2):
        if i+1 < X.shape[1]:
            # 避免除零
            ratio = X[:, i] / (X[:, i+1] + 1e-8)
            ratios.append(ratio.reshape(-1, 1))

    if ratios:
        ratios = np.hstack(ratios)
    else:
        ratios = np.array([]).reshape(X.shape[0], 0)

    # 3. 特征交互（选择性地创建）
    # 只创建前几个特征的交互，避免维度爆炸
    interactions = []
    for i in range(min(5, X.shape[1])):
        for j in range(i+1, min(5, X.shape[1])):
            interactions.append((X[:, i] * X[:, j]).reshape(-1, 1))

    if interactions:
        interactions = np.hstack(interactions)
    else:
        interactions = np.array([]).reshape(X.shape[0], 0)

    # 4. 多项式特征（平方项）- 只对前10个特征
    poly_features = []
    for i in range(min(10, X.shape[1])):
        poly_features.append((X[:, i] ** 2).reshape(-1, 1))

    poly_features = np.hstack(poly_features)

    # 合并所有特征
    X_enhanced = np.hstack([
        X_new,
        feature_mean,
        feature_std,
        feature_max,
        feature_min,
        ratios,
        interactions,
        poly_features
    ])

    return X_enhanced

# 创建增强特征
X_enhanced = create_enhanced_features(X)
print(f"增强后特征数: {X_enhanced.shape[1]}")

# ============================================================================
# 特征选择：使用LightGBM的特征重要性
# ============================================================================

print("\n" + "="*80)
print("特征选择")
print("="*80)

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(
    X_enhanced, y, test_size=0.2, random_state=42, stratify=y
)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练一个模型来获取特征重要性
print("\n训练特征选择模型...")
lgb_selector = LGBMClassifier(
    n_estimators=300,
    max_depth=15,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgb_selector.fit(X_train_scaled, y_train)

# 获取特征重要性
feature_importance = lgb_selector.feature_importances_
importance_threshold = np.percentile(feature_importance, 25)  # 保留前75%的特征

selected_features = feature_importance > importance_threshold
print(f"选择的特征数: {selected_features.sum()} / {len(selected_features)}")

# 使用选择的特征
X_train_selected = X_train_scaled[:, selected_features]
X_test_selected = X_test_scaled[:, selected_features]

# ============================================================================
# 模型训练与对比
# ============================================================================

print("\n" + "="*80)
print("模型性能对比")
print("="*80)

results = {}

# 1. 原始特征 + LightGBM
print("\n1. 测试原始特征...")
X_train_orig = scaler.fit_transform(X_train[:, :20])
X_test_orig = scaler.transform(X_test[:, :20])

lgb_orig = LGBMClassifier(
    n_estimators=500,
    max_depth=15,
    learning_rate=0.03,
    num_leaves=80,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgb_orig.fit(X_train_orig, y_train)
y_pred_orig = lgb_orig.predict(X_test_orig)
acc_orig = accuracy_score(y_test, y_pred_orig)
cv_orig = cross_val_score(lgb_orig, X_train_orig, y_train, cv=5, n_jobs=-1)

print(f"   原始特征准确率: {acc_orig:.4f} ({acc_orig*100:.2f}%)")
print(f"   交叉验证: {cv_orig.mean():.4f} (+/- {cv_orig.std():.4f})")
results['原始特征'] = acc_orig

# 2. 增强特征 + LightGBM
print("\n2. 测试增强特征...")
lgb_enhanced = LGBMClassifier(
    n_estimators=500,
    max_depth=15,
    learning_rate=0.03,
    num_leaves=80,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgb_enhanced.fit(X_train_scaled, y_train)
y_pred_enhanced = lgb_enhanced.predict(X_test_scaled)
acc_enhanced = accuracy_score(y_test, y_pred_enhanced)
cv_enhanced = cross_val_score(lgb_enhanced, X_train_scaled, y_train, cv=5, n_jobs=-1)

print(f"   增强特征准确率: {acc_enhanced:.4f} ({acc_enhanced*100:.2f}%)")
print(f"   交叉验证: {cv_enhanced.mean():.4f} (+/- {cv_enhanced.std():.4f})")
results['增强特征'] = acc_enhanced

# 3. 特征选择后 + LightGBM
print("\n3. 测试特征选择...")
lgb_selected = LGBMClassifier(
    n_estimators=500,
    max_depth=15,
    learning_rate=0.03,
    num_leaves=80,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgb_selected.fit(X_train_selected, y_train)
y_pred_selected = lgb_selected.predict(X_test_selected)
acc_selected = accuracy_score(y_test, y_pred_selected)
cv_selected = cross_val_score(lgb_selected, X_train_selected, y_train, cv=5, n_jobs=-1)

print(f"   特征选择准确率: {acc_selected:.4f} ({acc_selected*100:.2f}%)")
print(f"   交叉验证: {cv_selected.mean():.4f} (+/- {cv_selected.std():.4f})")
results['特征选择'] = acc_selected

# 4. 增强特征 + 优化超参数
print("\n4. 测试超参数优化...")
lgb_tuned = LGBMClassifier(
    n_estimators=1000,
    max_depth=20,
    learning_rate=0.02,
    num_leaves=100,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
lgb_tuned.fit(X_train_scaled, y_train)
y_pred_tuned = lgb_tuned.predict(X_test_scaled)
acc_tuned = accuracy_score(y_test, y_pred_tuned)
cv_tuned = cross_val_score(lgb_tuned, X_train_scaled, y_train, cv=5, n_jobs=-1)

print(f"   超参数优化准确率: {acc_tuned:.4f} ({acc_tuned*100:.2f}%)")
print(f"   交叉验证: {cv_tuned.mean():.4f} (+/- {cv_tuned.std():.4f})")
results['超参数优化'] = acc_tuned

# 5. 集成学习（使用最佳特征集）
print("\n5. 测试集成学习...")

# 选择最佳的特征集
best_X_train = X_train_scaled if acc_enhanced >= acc_selected else X_train_selected
best_X_test = X_test_scaled if acc_enhanced >= acc_selected else X_test_selected

lgb_ensemble = LGBMClassifier(
    n_estimators=1000,
    max_depth=20,
    learning_rate=0.02,
    num_leaves=100,
    min_child_samples=10,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

rf_ensemble = RandomForestClassifier(
    n_estimators=500,
    max_depth=35,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

et_ensemble = ExtraTreesClassifier(
    n_estimators=500,
    max_depth=35,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)

voting_clf = VotingClassifier(
    estimators=[
        ('lgb', lgb_ensemble),
        ('rf', rf_ensemble),
        ('et', et_ensemble)
    ],
    voting='soft',
    n_jobs=-1
)

voting_clf.fit(best_X_train, y_train)
y_pred_voting = voting_clf.predict(best_X_test)
acc_voting = accuracy_score(y_test, y_pred_voting)
cv_voting = cross_val_score(voting_clf, best_X_train, y_train, cv=5, n_jobs=-1)

print(f"   集成学习准确率: {acc_voting:.4f} ({acc_voting*100:.2f}%)")
print(f"   交叉验证: {cv_voting.mean():.4f} (+/- {cv_voting.std():.4f})")
results['集成学习'] = acc_voting

# ============================================================================
# 结果总结
# ============================================================================

print("\n" + "="*80)
print("结果总结")
print("="*80)

print("\n所有方法性能对比:")
for method, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    improvement = (acc - 0.8144) * 100
    print(f"  {method:12s}: {acc:.4f} ({acc*100:.2f}%) [相比基线 {improvement:+.2f}%]")

best_method = max(results, key=results.get)
best_acc = results[best_method]

print(f"\n最佳方法: {best_method}")
print(f"最佳准确率: {best_acc:.4f} ({best_acc*100:.2f}%)")
print(f"相比原始提升: {(best_acc - 0.8144)*100:.2f}%")

# 绘制最佳模型的混淆矩阵
print("\n绘制混淆矩阵...")

# 选择最佳模型
if best_method == '集成学习':
    best_model = voting_clf
    best_X_test_final = best_X_test
elif best_method == '超参数优化':
    best_model = lgb_tuned
    best_X_test_final = X_test_scaled
elif best_method == '增强特征':
    best_model = lgb_enhanced
    best_X_test_final = X_test_scaled
elif best_method == '特征选择':
    best_model = lgb_selected
    best_X_test_final = X_test_selected
else:
    best_model = lgb_orig
    best_X_test_final = X_test_orig

y_pred_best = best_model.predict(best_X_test_final)

cm = confusion_matrix(y_test, y_pred_best, labels=labels)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

# 计算TPR和FNR
tpr = np.diag(cm) / cm.sum(axis=1) * 100
fnr = 100 - tpr

# 创建带TPR/FNR的扩展矩阵
cm_extended = np.zeros((len(labels), len(labels) + 2))
cm_extended[:, :-2] = cm_percent
cm_extended[:, -2] = tpr
cm_extended[:, -1] = fnr

fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(cm_extended, annot=True, fmt='.1f', cmap='YlOrRd',
            xticklabels=list(labels) + ['TPR', 'FNR'],
            yticklabels=labels,
            cbar_kws={'label': '百分比 (%)'},
            linewidths=0.5, linecolor='white', ax=ax)
plt.title(f'{best_method}\n准确率: {best_acc*100:.2f}%',
          fontsize=13, pad=15)
plt.ylabel('真实类别', fontsize=11)
plt.xlabel('预测类别', fontsize=11)
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('improved_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("混淆矩阵已保存到 improved_confusion_matrix.png")

# 保存详细结果
with open('improvement_results.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("模型改进结果报告\n")
    f.write("="*80 + "\n\n")

    f.write(f"原始特征数: 20\n")
    f.write(f"增强后特征数: {X_enhanced.shape[1]}\n")
    f.write(f"选择后特征数: {selected_features.sum()}\n\n")

    f.write("性能对比:\n")
    f.write("-"*80 + "\n")
    for method, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        improvement = (acc - 0.8144) * 100
        f.write(f"{method}: {acc:.4f} ({acc*100:.2f}%) [提升 {improvement:+.2f}%]\n")

    f.write(f"\n最佳方法: {best_method}\n")
    f.write(f"最佳准确率: {best_acc:.4f} ({best_acc*100:.2f}%)\n")
    f.write(f"相比基线提升: {(best_acc - 0.8144)*100:.2f}%\n")

print("\n详细结果已保存到 improvement_results.txt")
