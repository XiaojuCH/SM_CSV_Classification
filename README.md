# 基于光学特征的多分类系统

基于 LightGBM 的光学特征分类系统，支持 Python 训练和 C++ 部署。

## 项目概述

本项目实现了一个六分类系统，用于识别不同类型的光学样本：

### 技术栈
- **训练**: Python + LightGBM + scikit-learn
- **部署**: C++ + ONNX Runtime
- **模型格式**: ONNX
- **准确率**: ~81.65%

---

## 项目结构

```
基于光学特征分类/
├── train_classifier.py          # 模型训练脚本
├── export_to_onnx.py            # ONNX 模型导出脚本
├── lightgbm_model.onnx          # ONNX 模型文件
├── scaler_params.json           # 特征标准化参数
├── label_mapping.json           # 类别映射表
├── confusion_matrix.png         # 混淆矩阵图
├── results.txt                  # 训练结果
├── DH.csv, KD.csv, ...          # 训练数据
├── TEST.csv                     # 测试数据
└── CPP/                         # C++ 部署项目
    ├── simple_classifier.cpp    # C++ 分类器源码
    ├── CMakeLists.txt           # CMake 构建配置
    ├── CMAKE使用指南.md         # CMake 使用说明
    └── build/                   # 构建目录
        └── bin/Release/         # 可执行文件目录
            ├── SimpleClassifier.exe
            ├── onnxruntime.dll
            ├── lightgbm_model.onnx
            ├── scaler_params.json
            └── label_mapping.json
```

---

## 快速开始

### 1. Python 训练模型

#### 环境要求
```bash
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn
```

#### 训练模型
```bash
python train_classifier.py
```

**输出文件：**
- `lightgbm_model.pkl` - Python 模型
- `scaler.pkl` - 标准化器
- `confusion_matrix.png` - 混淆矩阵
- `results.txt` - 训练结果

---

### 2. 导出 ONNX 模型

#### 环境要求
```bash
pip install onnxmltools onnxruntime
```

#### 导出模型
```bash
python export_to_onnx.py
```

**输出文件：**
- `lightgbm_model.onnx` - ONNX 模型
- `scaler_params.json` - 标准化参数
- `label_mapping.json` - 类别映射

---

### 3. C++ 部署

#### 环境要求
- **CMake** 3.15+
- **Visual Studio 2019+** (Windows)
- **ONNX Runtime** 1.19.2+

#### 安装 ONNX Runtime

1. 下载最新版本：https://github.com/microsoft/onnxruntime/releases
2. 解压到 `C:\onnxruntime`（或其他目录）

#### 编译项目

```bash
cd CPP

# 配置 CMake
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DONNXRUNTIME_DIR=C:/onnxruntime

# 编译
cmake --build build --config Release
```

#### 运行程序

```bash
cd build\bin\Release

# 预测单个 CSV 文件
.\SimpleClassifier.exe path\to\your\data.csv

# 示例
.\SimpleClassifier.exe D:\Projects_\Qinghua_Project\基于光学特征分类\TEST.csv
```

---

## 使用说明

### CSV 数据格式

输入的 CSV 文件必须满足以下要求：
- **每行 20 个特征值**
- **用逗号分隔**
- **无表头行**
- **纯数字数据**

**示例：**
```csv
581.722,-0.411162,-0.262966,0.1029,355.387,...
454.773,-0.31918,-0.311743,0.148278,469.285,...
```

### 特征说明

每个样本包含 20 个光学特征：
- 特征 1-4: I, Q, U, V 参数
- 特征 5-20: 缪勒矩阵相关参数

---

## 输出示例

```
============================================================
LightGBM ONNX Classifier - Batch Prediction
============================================================

Input CSV file: TEST.csv

[1/4] Loading scaler parameters...
  Number of features: 20

[2/4] Loading label mapping...
  Number of classes: 6
  Classes: DH KD PS10 PS10-H QZ YM

[3/4] Loading ONNX model...
  Model loaded successfully!

[4/5] Reading CSV data...
  Total samples: 2

[5/5] Starting batch prediction...
============================================================

[Sample 1]
  Features (first 5): 581.722 -0.411162 -0.262966 0.1029 355.387 ...
  Probabilities: DH=0.00% KD=0.00% PS10=0.00% PS10-H=0.00% QZ=0.00% YM=99.99%
  Prediction: YM (Confidence: 99.99%)

[Sample 2]
  Features (first 5): 454.773 -0.31918 -0.311743 0.148278 469.285 ...
  Probabilities: DH=99.99% KD=0.00% PS10=0.00% PS10-H=0.00% QZ=0.00% YM=0.00%
  Prediction: DH (Confidence: 99.99%)

============================================================
Batch prediction completed!
Total samples processed: 2
============================================================
```

---

## 性能指标

### 模型性能
- **测试准确率**: 81.65%
- **交叉验证准确率**: 81.64% (±0.0044)
- **模型大小**: 28.28 MB
- **特征数量**: 20
- **类别数量**: 6

### C++ 运行性能
- **模型加载时间**: < 1 秒
- **单样本预测**: < 1 毫秒
- **批量预测**: 支持

---

## 常见问题

### Q1: 模型加载很慢（超过 1 分钟）
**A:** 这是 ONNX Runtime 版本过旧导致的。请升级到最新版本（1.19.2+）。

### Q2: 编译时找不到 ONNX Runtime
**A:** 检查 CMake 配置中的 `ONNXRUNTIME_DIR` 路径是否正确。

### Q3: 运行时提示缺少 DLL
**A:** 确保 `onnxruntime.dll` 与可执行文件在同一目录。

### Q4: CSV 文件读取失败
**A:** 检查 CSV 格式：
- 确保每行有 20 个特征
- 确保用逗号分隔
- 确保没有表头行
- 确保是纯文本格式（不是 Excel 文件）

### Q5: 预测结果不准确
**A:** 确保：
- 使用正确的模型文件
- 特征顺序与训练时一致
- 数据已正确标准化

---

## 技术细节

### 模型架构
- **算法**: LightGBM (Gradient Boosting)
- **树数量**: 1000
- **最大深度**: 15
- **学习率**: 0.1
- **叶子节点数**: 80

### 数据预处理
- **标准化**: StandardScaler (零均值，单位方差)
- **公式**: `(x - mean) / scale`

### ONNX 导出配置
- **Opset 版本**: 12
- **输出格式**: 概率数组 (zipmap=False)
- **输入形状**: [batch_size, 20]
- **输出形状**: [batch_size, 6]

---

## 开发者信息

### 重新训练模型

如果需要重新训练模型：

1. 准备数据（CSV 格式，每个类别一个文件）
2. 修改 `train_classifier.py` 中的文件列表
3. 运行训练脚本
4. 导出 ONNX 模型
5. 重新编译 C++ 项目

### 修改类别

如果需要修改类别数量或名称：

1. 修改 `train_classifier.py` 中的 `labels` 列表
2. 重新训练模型
3. 导出 ONNX 模型
4. C++ 代码会自动适配（从 `label_mapping.json` 读取）

### 添加新特征

如果需要添加新特征：

1. 修改数据文件（增加特征列）
2. 修改 `export_to_onnx.py` 中的特征数量
3. 修改 `simple_classifier.cpp` 中的特征数量（第 20 行）
4. 重新训练和导出模型

---

## 许可证

本项目遵循 MIT 许可证。

---

## 更新日志

### v1.0.0 (2025-12-06)
- ✅ 实现 LightGBM 模型训练
- ✅ 实现 ONNX 模型导出
- ✅ 实现 C++ 分类器
- ✅ 支持批量预测
- ✅ 优化模型加载速度（430倍提升）

---

## 联系方式

如有问题或建议，请提交 Issue。
