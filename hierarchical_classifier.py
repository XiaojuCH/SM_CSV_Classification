import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
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
print("层次分类策略")
print("="*80)
print("\n策略说明:")
print("基于混淆矩阵分析，我们发现以下类别对最容易混淆:")
print("  1. DH vs YM (距离: 0.86)")
print("  2. PS10 vs PS10-H (混淆率: 14.45%)")
print("  3. QZ vs YM (互相混淆)")
print("\n层次分类策略:")
print("  第一层: 将6类分为3个超类")
print("    - 超类A: DH + YM (容易混淆)")
print("    - 超类B: PS10 + PS10-H (容易混淆)")
print("    - 超类C: KD + QZ (相对独立)")
print("  第二层: 在每个超类内部进行细分类")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# 方法1: 基础层次分类
# ============================================================================

print("\n" + "="*80)
print("方法1: 基础层次分类")
print("="*80)

# 创建超类标签
def create_superclass_labels(y):
    """创建超类标签"""
    y_super = np.empty_like(y, dtype=object)
    for i, label in enumerate(y):
        if label in ['DH', 'YM']:
            y_super[i] = 'A_DH_YM'
        elif label in ['PS10', 'PS10-H']:
            y_super[i] = 'B_PS10'
        else:  # KD, QZ
            y_super[i] = 'C_KD_QZ'
    return y_super

y_train_super = create_superclass_labels(y_train)
y_test_super = create_superclass_labels(y_test)

# 第一层: 超类分类器
print("\n训练第一层分类器 (超类分类)...")
clf_level1 = LGBMClassifier(
    n_estimators=500,
    max_depth=15,
    learning_rate=0.03,
    num_leaves=80,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
clf_level1.fit(X_train_scaled, y_train_super)
y_pred_super = clf_level1.predict(X_test_scaled)
acc_super = accuracy_score(y_test_super, y_pred_super)
print(f"第一层准确率: {acc_super:.4f} ({acc_super*100:.2f}%)")

# 第二层: 为每个超类训练专门的分类器
print("\n训练第二层分类器 (细分类)...")

# 超类A: DH vs YM
mask_A_train = np.isin(y_train, ['DH', 'YM'])
mask_A_test = np.isin(y_test, ['DH', 'YM'])

clf_A = LGBMClassifier(
    n_estimators=800,
    max_depth=20,
    learning_rate=0.02,
    num_leaves=100,
    min_child_samples=5,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.2,
    reg_lambda=0.2,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
clf_A.fit(X_train_scaled[mask_A_train], y_train[mask_A_train])
acc_A = accuracy_score(y_test[mask_A_test], clf_A.predict(X_test_scaled[mask_A_test]))
print(f"  超类A (DH vs YM) 准确率: {acc_A:.4f} ({acc_A*100:.2f}%)")

# 超类B: PS10 vs PS10-H
mask_B_train = np.isin(y_train, ['PS10', 'PS10-H'])
mask_B_test = np.isin(y_test, ['PS10', 'PS10-H'])

clf_B = LGBMClassifier(
    n_estimators=800,
    max_depth=20,
    learning_rate=0.02,
    num_leaves=100,
    min_child_samples=5,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.2,
    reg_lambda=0.2,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
clf_B.fit(X_train_scaled[mask_B_train], y_train[mask_B_train])
acc_B = accuracy_score(y_test[mask_B_test], clf_B.predict(X_test_scaled[mask_B_test]))
print(f"  超类B (PS10 vs PS10-H) 准确率: {acc_B:.4f} ({acc_B*100:.2f}%)")

# 超类C: KD vs QZ
mask_C_train = np.isin(y_train, ['KD', 'QZ'])
mask_C_test = np.isin(y_test, ['KD', 'QZ'])

clf_C = LGBMClassifier(
    n_estimators=800,
    max_depth=20,
    learning_rate=0.02,
    num_leaves=100,
    min_child_samples=5,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.2,
    reg_lambda=0.2,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
clf_C.fit(X_train_scaled[mask_C_train], y_train[mask_C_train])
acc_C = accuracy_score(y_test[mask_C_test], clf_C.predict(X_test_scaled[mask_C_test]))
print(f"  超类C (KD vs QZ) 准确率: {acc_C:.4f} ({acc_C*100:.2f}%)")

# 组合预测
print("\n组合两层预测结果...")
y_pred_hierarchical = np.empty_like(y_test, dtype=object)

for i in range(len(X_test_scaled)):
    super_pred = y_pred_super[i]

    if super_pred == 'A_DH_YM':
        y_pred_hierarchical[i] = clf_A.predict(X_test_scaled[i:i+1])[0]
    elif super_pred == 'B_PS10':
        y_pred_hierarchical[i] = clf_B.predict(X_test_scaled[i:i+1])[0]
    else:  # C_KD_QZ
        y_pred_hierarchical[i] = clf_C.predict(X_test_scaled[i:i+1])[0]

acc_hierarchical = accuracy_score(y_test, y_pred_hierarchical)
print(f"\n层次分类总体准确率: {acc_hierarchical:.4f} ({acc_hierarchical*100:.2f}%)")

# ============================================================================
# 方法2: 改进的层次分类 (One-vs-Rest策略)
# ============================================================================

print("\n" + "="*80)
print("方法2: 改进的层次分类 (针对混淆类别对)")
print("="*80)

# 策略: 先识别容易分类的，再处理难分类的
print("\n训练策略:")
print("  步骤1: 先识别最容易的类别 (KD)")
print("  步骤2: 识别 PS10")
print("  步骤3: 在剩余样本中区分 PS10-H")
print("  步骤4: 在剩余样本中区分 QZ")
print("  步骤5: 在剩余样本中区分 DH vs YM")

# 步骤1: KD vs 其他
print("\n步骤1: 识别 KD...")
y_train_kd = (y_train == 'KD').astype(int)
clf_kd = LGBMClassifier(
    n_estimators=500,
    max_depth=15,
    learning_rate=0.03,
    num_leaves=80,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
clf_kd.fit(X_train_scaled, y_train_kd)
prob_kd = clf_kd.predict_proba(X_test_scaled)[:, 1]

# 步骤2: PS10 vs 其他
print("步骤2: 识别 PS10...")
y_train_ps10 = (y_train == 'PS10').astype(int)
clf_ps10 = LGBMClassifier(
    n_estimators=500,
    max_depth=15,
    learning_rate=0.03,
    num_leaves=80,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
clf_ps10.fit(X_train_scaled, y_train_ps10)
prob_ps10 = clf_ps10.predict_proba(X_test_scaled)[:, 1]

# 步骤3: PS10-H vs 其他
print("步骤3: 识别 PS10-H...")
y_train_ps10h = (y_train == 'PS10-H').astype(int)
clf_ps10h = LGBMClassifier(
    n_estimators=500,
    max_depth=15,
    learning_rate=0.03,
    num_leaves=80,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
clf_ps10h.fit(X_train_scaled, y_train_ps10h)
prob_ps10h = clf_ps10h.predict_proba(X_test_scaled)[:, 1]

# 步骤4: QZ vs 其他
print("步骤4: 识别 QZ...")
y_train_qz = (y_train == 'QZ').astype(int)
clf_qz = LGBMClassifier(
    n_estimators=500,
    max_depth=15,
    learning_rate=0.03,
    num_leaves=80,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
clf_qz.fit(X_train_scaled, y_train_qz)
prob_qz = clf_qz.predict_proba(X_test_scaled)[:, 1]

# 步骤5: DH vs YM (在剩余样本中)
print("步骤5: 区分 DH vs YM...")
mask_dh_ym_train = np.isin(y_train, ['DH', 'YM'])
clf_dh_ym = LGBMClassifier(
    n_estimators=800,
    max_depth=20,
    learning_rate=0.02,
    num_leaves=100,
    min_child_samples=5,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.2,
    reg_lambda=0.2,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
clf_dh_ym.fit(X_train_scaled[mask_dh_ym_train], y_train[mask_dh_ym_train])
prob_dh_ym = clf_dh_ym.predict_proba(X_test_scaled)

# 组合所有概率进行预测
print("\n组合所有分类器的预测...")
y_pred_ovr = np.empty_like(y_test, dtype=object)

for i in range(len(X_test_scaled)):
    # 收集所有类别的概率
    probs = {
        'KD': prob_kd[i],
        'PS10': prob_ps10[i],
        'PS10-H': prob_ps10h[i],
        'QZ': prob_qz[i]
    }

    # 对于DH和YM，使用专门的二分类器
    dh_ym_probs = prob_dh_ym[i]
    # 找到DH和YM在clf_dh_ym中的索引
    dh_idx = list(clf_dh_ym.classes_).index('DH')
    ym_idx = list(clf_dh_ym.classes_).index('YM')
    probs['DH'] = dh_ym_probs[dh_idx]
    probs['YM'] = dh_ym_probs[ym_idx]

    # 选择概率最高的类别
    y_pred_ovr[i] = max(probs, key=probs.get)

acc_ovr = accuracy_score(y_test, y_pred_ovr)
print(f"\nOne-vs-Rest策略准确率: {acc_ovr:.4f} ({acc_ovr*100:.2f}%)")

# ============================================================================
# 方法3: 混合策略 (结合原始模型和层次分类)
# ============================================================================

print("\n" + "="*80)
print("方法3: 混合策略")
print("="*80)

# 训练原始的全局分类器
print("\n训练全局分类器...")
clf_global = LGBMClassifier(
    n_estimators=500,
    max_depth=15,
    learning_rate=0.03,
    num_leaves=80,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
clf_global.fit(X_train_scaled, y_train)
prob_global = clf_global.predict_proba(X_test_scaled)
y_pred_global = clf_global.predict(X_test_scaled)

# 对于全局分类器不确定的样本，使用专门的分类器
print("对不确定样本使用专门分类器...")
y_pred_hybrid = y_pred_global.copy()

for i in range(len(X_test_scaled)):
    max_prob = prob_global[i].max()
    pred_class = y_pred_global[i]

    # 如果预测概率低于阈值，使用专门的分类器
    if max_prob < 0.6:  # 不确定阈值
        # 如果预测为DH或YM，使用专门的DH-YM分类器
        if pred_class in ['DH', 'YM']:
            y_pred_hybrid[i] = clf_A.predict(X_test_scaled[i:i+1])[0]
        # 如果预测为PS10或PS10-H，使用专门的PS10分类器
        elif pred_class in ['PS10', 'PS10-H']:
            y_pred_hybrid[i] = clf_B.predict(X_test_scaled[i:i+1])[0]
        # 如果预测为KD或QZ，使用专门的KD-QZ分类器
        elif pred_class in ['KD', 'QZ']:
            y_pred_hybrid[i] = clf_C.predict(X_test_scaled[i:i+1])[0]

acc_hybrid = accuracy_score(y_test, y_pred_hybrid)
print(f"\n混合策略准确率: {acc_hybrid:.4f} ({acc_hybrid*100:.2f}%)")

# ============================================================================
# 结果对比
# ============================================================================

print("\n" + "="*80)
print("结果总结")
print("="*80)

results = {
    '原始LightGBM': 0.8144,
    '基础层次分类': acc_hierarchical,
    'One-vs-Rest策略': acc_ovr,
    '混合策略': acc_hybrid
}

print("\n所有方法性能对比:")
for method, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
    improvement = (acc - 0.8144) * 100
    print(f"  {method:20s}: {acc:.4f} ({acc*100:.2f}%) [相比基线 {improvement:+.2f}%]")

best_method = max(results, key=results.get)
best_acc = results[best_method]

print(f"\n最佳方法: {best_method}")
print(f"最佳准确率: {best_acc:.4f} ({best_acc*100:.2f}%)")
print(f"相比原始提升: {(best_acc - 0.8144)*100:.2f}%")

# 选择最佳预测结果
if best_method == '基础层次分类':
    y_pred_best = y_pred_hierarchical
elif best_method == 'One-vs-Rest策略':
    y_pred_best = y_pred_ovr
elif best_method == '混合策略':
    y_pred_best = y_pred_hybrid
else:
    y_pred_best = y_pred_global

# 绘制混淆矩阵
print("\n绘制最佳方法的混淆矩阵...")
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
plt.savefig('hierarchical_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("混淆矩阵已保存到 hierarchical_confusion_matrix.png")

# 详细的分类报告
print("\n" + "="*80)
print("详细分类报告")
print("="*80)
print(classification_report(y_test, y_pred_best, target_names=labels))

# 保存结果
with open('hierarchical_results.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("层次分类策略结果报告\n")
    f.write("="*80 + "\n\n")

    f.write("策略说明:\n")
    f.write("  方法1: 基础层次分类 - 先分超类，再细分\n")
    f.write("  方法2: One-vs-Rest策略 - 逐个识别每个类别\n")
    f.write("  方法3: 混合策略 - 全局+专门分类器\n\n")

    f.write("性能对比:\n")
    f.write("-"*80 + "\n")
    for method, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        improvement = (acc - 0.8144) * 100
        f.write(f"{method}: {acc:.4f} ({acc*100:.2f}%) [提升 {improvement:+.2f}%]\n")

    f.write(f"\n最佳方法: {best_method}\n")
    f.write(f"最佳准确率: {best_acc:.4f} ({best_acc*100:.2f}%)\n")
    f.write(f"相比基线提升: {(best_acc - 0.8144)*100:.2f}%\n\n")

    f.write("详细分类报告:\n")
    f.write("-"*80 + "\n")
    f.write(classification_report(y_test, y_pred_best, target_names=labels))

print("\n详细结果已保存到 hierarchical_results.txt")
