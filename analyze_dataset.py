import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
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
print("数据集基本信息分析")
print("="*80)
print(f"\n总样本数: {len(X)}")
print(f"特征维度: {X.shape[1]}")
print(f"类别数量: {len(labels)}")
print(f"\n类别分布:")
for label in labels:
    count = np.sum(y == label)
    print(f"  {label}: {count:5d} ({count/len(y)*100:5.2f}%)")

# 检查类别不平衡程度
class_counts = [np.sum(y == label) for label in labels]
imbalance_ratio = max(class_counts) / min(class_counts)
print(f"\n类别不平衡比: {imbalance_ratio:.2f}:1 (最大类/最小类)")

# 特征统计分析
print("\n" + "="*80)
print("特征统计分析")
print("="*80)
print(f"特征均值范围: [{X.mean(axis=0).min():.2f}, {X.mean(axis=0).max():.2f}]")
print(f"特征标准差范围: [{X.std(axis=0).min():.2f}, {X.std(axis=0).max():.2f}]")
print(f"特征最小值: {X.min():.2f}")
print(f"特征最大值: {X.max():.2f}")

# 检查缺失值和异常值
print(f"\n缺失值数量: {np.isnan(X).sum()}")
print(f"无穷值数量: {np.isinf(X).sum()}")

# 类间距离分析
print("\n" + "="*80)
print("类间可分性分析")
print("="*80)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 计算每个类别的中心
class_centers = []
for label in labels:
    class_mask = (y == label)
    class_center = X_scaled[class_mask].mean(axis=0)
    class_centers.append(class_center)

class_centers = np.array(class_centers)

# 计算类间距离矩阵
distances = cdist(class_centers, class_centers, metric='euclidean')
print("\n类间中心距离矩阵:")
print("     ", "  ".join([f"{l:>6s}" for l in labels]))
for i, label in enumerate(labels):
    print(f"{label:>6s}", "  ".join([f"{d:6.2f}" for d in distances[i]]))

# 找出最难区分的类别对
min_dist = float('inf')
min_pair = None
for i in range(len(labels)):
    for j in range(i+1, len(labels)):
        if distances[i][j] < min_dist:
            min_dist = distances[i][j]
            min_pair = (labels[i], labels[j])

print(f"\n最难区分的类别对: {min_pair[0]} vs {min_pair[1]} (距离: {min_dist:.2f})")

# 计算类内方差
print("\n类内方差分析:")
for label in labels:
    class_mask = (y == label)
    class_data = X_scaled[class_mask]
    intra_var = np.mean(np.var(class_data, axis=0))
    print(f"  {label}: {intra_var:.4f}")

# 使用多个高级模型进行集成测试
print("\n" + "="*80)
print("高级模型性能测试")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 测试多个模型
models = {
    'LightGBM (优化)': LGBMClassifier(
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
    ),
    'ExtraTrees': ExtraTreesClassifier(
        n_estimators=500,
        max_depth=35,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    ),
    'Random Forest (优化)': RandomForestClassifier(
        n_estimators=500,
        max_depth=35,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
}

results = {}
best_models = {}

for name, model in models.items():
    print(f"\n训练 {name}...")
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)

    # 5折交叉验证
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, n_jobs=-1)

    results[name] = {
        'accuracy': accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'cv_scores': cv_scores
    }
    best_models[name] = model

    print(f"  测试准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  交叉验证: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# 集成学习
print("\n" + "="*80)
print("集成学习测试")
print("="*80)

# 软投票集成
voting_clf = VotingClassifier(
    estimators=[
        ('lgb', best_models['LightGBM (优化)']),
        ('et', best_models['ExtraTrees']),
        ('rf', best_models['Random Forest (优化)'])
    ],
    voting='soft',
    n_jobs=-1
)

print("\n训练集成模型 (软投票)...")
voting_clf.fit(X_train_scaled, y_train)
y_pred_voting = voting_clf.predict(X_test_scaled)
voting_accuracy = accuracy_score(y_test, y_pred_voting)

cv_voting = cross_val_score(voting_clf, X_train_scaled, y_train, cv=5, n_jobs=-1)

print(f"集成模型测试准确率: {voting_accuracy:.4f} ({voting_accuracy*100:.2f}%)")
print(f"集成模型交叉验证: {cv_voting.mean():.4f} (+/- {cv_voting.std():.4f})")

results['集成模型 (软投票)'] = {
    'accuracy': voting_accuracy,
    'cv_mean': cv_voting.mean(),
    'cv_std': cv_voting.std()
}

# 分析混淆矩阵找出难分类的样本
print("\n" + "="*80)
print("错误分类分析")
print("="*80)

best_single_model = max(results.items(), key=lambda x: x[1]['accuracy'])
print(f"\n最佳单模型: {best_single_model[0]}")
print(f"准确率: {best_single_model[1]['accuracy']:.4f} ({best_single_model[1]['accuracy']*100:.2f}%)")

best_model = best_models[best_single_model[0]]
y_pred_best = best_model.predict(X_test_scaled)

cm = confusion_matrix(y_test, y_pred_best, labels=labels)
print("\n混淆矩阵:")
print("真实\\预测", "  ".join([f"{l:>6s}" for l in labels]))
for i, label in enumerate(labels):
    print(f"{label:>6s}   ", "  ".join([f"{c:6d}" for c in cm[i]]))

# 计算每个类别的准确率
print("\n各类别准确率:")
for i, label in enumerate(labels):
    class_acc = cm[i][i] / cm[i].sum() * 100
    print(f"  {label}: {class_acc:.2f}%")

# 找出最容易混淆的类别对
print("\n最容易混淆的类别对 (非对角线最大值):")
confusion_pairs = []
for i in range(len(labels)):
    for j in range(len(labels)):
        if i != j and cm[i][j] > 0:
            confusion_pairs.append((labels[i], labels[j], cm[i][j], cm[i][j]/cm[i].sum()*100))

confusion_pairs.sort(key=lambda x: x[3], reverse=True)
for true_label, pred_label, count, percent in confusion_pairs[:5]:
    print(f"  {true_label} -> {pred_label}: {count} 次 ({percent:.2f}%)")

# 估算理论上限
print("\n" + "="*80)
print("数据集性能上限估算")
print("="*80)

# 使用贝叶斯错误率估算
# 通过最近邻分析估算
from sklearn.neighbors import KNeighborsClassifier

knn_scores = []
for k in [1, 3, 5, 7, 9]:
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    cv_knn = cross_val_score(knn, X_train_scaled, y_train, cv=5, n_jobs=-1)
    knn_scores.append(cv_knn.mean())
    print(f"KNN (k={k}) 交叉验证准确率: {cv_knn.mean():.4f} ({cv_knn.mean()*100:.2f}%)")

# 理论上限估算
print("\n基于多种方法的性能上限估算:")
print(f"  1-NN (贝叶斯错误率近似): {knn_scores[0]*100:.2f}%")
print(f"  最佳单模型: {best_single_model[1]['accuracy']*100:.2f}%")
print(f"  集成模型: {voting_accuracy*100:.2f}%")
print(f"  估算理论上限: {max(knn_scores[0], voting_accuracy)*100:.2f}% - {min(95, max(knn_scores[0], voting_accuracy)*100 + 5):.2f}%")

# 总结和建议
print("\n" + "="*80)
print("总结与建议")
print("="*80)

current_best = 81.44
improvement_potential = max(voting_accuracy*100, best_single_model[1]['accuracy']*100) - current_best

print(f"\n当前最佳性能: {current_best:.2f}%")
print(f"本次测试最佳: {max(voting_accuracy*100, best_single_model[1]['accuracy']*100):.2f}%")
print(f"潜在提升空间: {improvement_potential:.2f}%")

print("\n改进建议:")
print("1. 特征工程:")
print("   - 尝试特征交互 (多项式特征)")
print("   - 特征选择 (去除冗余特征)")
print("   - 特征变换 (对数、平方根等)")

print("\n2. 模型优化:")
print("   - 使用贝叶斯优化进行超参数调优")
print("   - 尝试深度学习模型 (MLP, 1D-CNN)")
print("   - 使用Stacking集成方法")

print("\n3. 数据增强:")
print("   - 对少数类进行过采样 (SMOTE)")
print("   - 数据清洗 (去除噪声样本)")

print(f"\n4. 类别特定优化:")
print(f"   - 重点关注混淆度高的类别对: {min_pair[0]} vs {min_pair[1]}")
print("   - 考虑使用层次分类策略")

# 保存详细结果
with open('detailed_analysis.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("数据集深度分析报告\n")
    f.write("="*80 + "\n\n")

    f.write(f"数据集规模: {len(X)} 样本, {X.shape[1]} 特征, {len(labels)} 类别\n")
    f.write(f"类别不平衡比: {imbalance_ratio:.2f}:1\n\n")

    f.write("模型性能对比:\n")
    f.write("-"*80 + "\n")
    for name, result in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        f.write(f"{name}:\n")
        f.write(f"  测试准确率: {result['accuracy']:.4f} ({result['accuracy']*100:.2f}%)\n")
        f.write(f"  交叉验证: {result['cv_mean']:.4f} (+/- {result['cv_std']:.4f})\n\n")

    f.write(f"\n当前最佳: {current_best:.2f}%\n")
    f.write(f"本次最佳: {max(voting_accuracy*100, best_single_model[1]['accuracy']*100):.2f}%\n")
    f.write(f"估算上限: {max(knn_scores[0], voting_accuracy)*100:.2f}% - {min(95, max(knn_scores[0], voting_accuracy)*100 + 5):.2f}%\n")

print("\n详细分析已保存到 detailed_analysis.txt")
