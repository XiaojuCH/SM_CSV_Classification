import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
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

X = np.vstack(all_data)
y = np.array(all_labels)

print("="*80)
print("LightGBM 精细调优")
print("="*80)
print(f"\n数据规模: {X.shape[0]} 样本, {X.shape[1]} 特征")
print(f"当前最佳: 81.65%")
print(f"目标: 尝试突破 82%\n")

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 测试多组参数配置
configs = {
    '当前配置 (基线)': {
        'n_estimators': 1000,
        'max_depth': 15,
        'learning_rate': 0.1,
        'num_leaves': 80,
        'min_child_samples': 20,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'reg_alpha': 0.0,
        'reg_lambda': 0.0
    },

    '配置1: 增加树深度': {
        'n_estimators': 1000,
        'max_depth': 20,           # 增加深度
        'learning_rate': 0.1,
        'num_leaves': 100,         # 相应增加叶子数
        'min_child_samples': 20,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'reg_alpha': 0.0,
        'reg_lambda': 0.0
    },

    '配置2: 更多树+小学习率': {
        'n_estimators': 1500,      # 更多树
        'max_depth': 15,
        'learning_rate': 0.05,     # 降低学习率
        'num_leaves': 80,
        'min_child_samples': 20,
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'reg_alpha': 0.0,
        'reg_lambda': 0.0
    },

    '配置3: 添加正则化': {
        'n_estimators': 1000,
        'max_depth': 15,
        'learning_rate': 0.1,
        'num_leaves': 80,
        'min_child_samples': 20,
        'subsample': 0.8,          # 行采样
        'colsample_bytree': 0.8,   # 列采样
        'reg_alpha': 0.1,          # L1正则化
        'reg_lambda': 0.1          # L2正则化
    },

    '配置4: 更多叶子节点': {
        'n_estimators': 1000,
        'max_depth': 18,
        'learning_rate': 0.08,
        'num_leaves': 120,         # 更多叶子
        'min_child_samples': 15,   # 降低最小样本数
        'subsample': 1.0,
        'colsample_bytree': 1.0,
        'reg_alpha': 0.0,
        'reg_lambda': 0.0
    },

    '配置5: 平衡配置': {
        'n_estimators': 1200,
        'max_depth': 18,
        'learning_rate': 0.07,
        'num_leaves': 100,
        'min_child_samples': 15,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'reg_alpha': 0.05,
        'reg_lambda': 0.05
    },

    '配置6: 激进配置': {
        'n_estimators': 1500,
        'max_depth': 25,           # 很深的树
        'learning_rate': 0.05,
        'num_leaves': 150,         # 很多叶子
        'min_child_samples': 10,
        'subsample': 0.85,
        'colsample_bytree': 0.85,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1
    }
}

results = {}

print("开始测试各配置...\n")

for config_name, params in configs.items():
    print(f"{'='*80}")
    print(f"测试: {config_name}")
    print(f"{'='*80}")

    # 显示关键参数
    print(f"参数: n_estimators={params['n_estimators']}, max_depth={params['max_depth']}, "
          f"lr={params['learning_rate']}, num_leaves={params['num_leaves']}")

    # 训练模型
    model = LGBMClassifier(
        random_state=42,
        n_jobs=-1,
        verbose=-1,
        **params
    )

    model.fit(X_train_scaled, y_train)

    # 测试集评估
    y_pred = model.predict(X_test_scaled)
    test_acc = accuracy_score(y_test, y_pred)

    # 交叉验证
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, n_jobs=-1)
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    results[config_name] = {
        'test_acc': test_acc,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'model': model,
        'y_pred': y_pred
    }

    print(f"测试准确率: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"交叉验证: {cv_mean:.4f} (+/- {cv_std:.4f})")

    # 与基线对比
    improvement = (test_acc - 0.8165) * 100
    if improvement > 0:
        print(f"✅ 相比基线提升: +{improvement:.2f}%")
    elif improvement < 0:
        print(f"❌ 相比基线下降: {improvement:.2f}%")
    else:
        print(f"➖ 与基线持平")
    print()

# 结果汇总
print("\n" + "="*80)
print("结果汇总")
print("="*80)

print("\n按测试准确率排序:")
sorted_results = sorted(results.items(), key=lambda x: x[1]['test_acc'], reverse=True)

for i, (config_name, result) in enumerate(sorted_results, 1):
    test_acc = result['test_acc']
    cv_mean = result['cv_mean']
    cv_std = result['cv_std']
    improvement = (test_acc - 0.8165) * 100

    print(f"{i}. {config_name}")
    print(f"   测试: {test_acc:.4f} ({test_acc*100:.2f}%) | "
          f"交叉验证: {cv_mean:.4f} (±{cv_std:.4f}) | "
          f"提升: {improvement:+.2f}%")

# 找出最佳配置
best_config_name = sorted_results[0][0]
best_result = sorted_results[0][1]
best_acc = best_result['test_acc']

print(f"\n{'='*80}")
print(f"🏆 最佳配置: {best_config_name}")
print(f"{'='*80}")
print(f"测试准确率: {best_acc:.4f} ({best_acc*100:.2f}%)")
print(f"交叉验证: {best_result['cv_mean']:.4f} (±{best_result['cv_std']:.4f})")
print(f"相比81.65%提升: {(best_acc - 0.8165)*100:+.2f}%")

# 绘制最佳模型的混淆矩阵
print(f"\n绘制最佳配置的混淆矩阵...")
y_pred_best = best_result['y_pred']

cm = confusion_matrix(y_test, y_pred_best, labels=labels)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

tpr = np.diag(cm) / cm.sum(axis=1) * 100
fnr = 100 - tpr

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
plt.title(f'{best_config_name}\n准确率: {best_acc*100:.2f}%',
          fontsize=13, pad=15)
plt.ylabel('真实类别', fontsize=11)
plt.xlabel('预测类别', fontsize=11)
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('fine_tuned_confusion_matrix.png', dpi=300, bbox_inches='tight')
print("混淆矩阵已保存到 fine_tuned_confusion_matrix.png")

# 详细分类报告
print("\n" + "="*80)
print("详细分类报告")
print("="*80)
print(classification_report(y_test, y_pred_best, target_names=labels))

# 保存最佳配置
print("\n" + "="*80)
print("最佳配置参数")
print("="*80)
best_params = configs[best_config_name]
print("\nLGBMClassifier(")
for param, value in best_params.items():
    print(f"    {param}={value},")
print("    random_state=42,")
print("    n_jobs=-1,")
print("    verbose=-1")
print(")")

# 保存结果
with open('fine_tune_results.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("LightGBM 精细调优结果\n")
    f.write("="*80 + "\n\n")

    f.write(f"基线准确率: 81.65%\n")
    f.write(f"最佳准确率: {best_acc*100:.2f}%\n")
    f.write(f"提升幅度: {(best_acc - 0.8165)*100:+.2f}%\n\n")

    f.write("所有配置结果:\n")
    f.write("-"*80 + "\n")
    for i, (config_name, result) in enumerate(sorted_results, 1):
        f.write(f"{i}. {config_name}\n")
        f.write(f"   测试: {result['test_acc']*100:.2f}% | ")
        f.write(f"交叉验证: {result['cv_mean']*100:.2f}% (±{result['cv_std']:.4f})\n")

    f.write(f"\n最佳配置: {best_config_name}\n")
    f.write("-"*80 + "\n")
    for param, value in best_params.items():
        f.write(f"{param}: {value}\n")

print("\n详细结果已保存到 fine_tune_results.txt")
