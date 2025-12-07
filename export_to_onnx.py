"""
将训练好的LightGBM模型导出为ONNX格式
用于C++部署
"""
import joblib
import numpy as np
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import json

print("="*60)
print("导出模型为ONNX格式")
print("="*60)

# 1. 加载训练好的模型和scaler
print("\n加载模型文件...")
try:
    model = joblib.load('lightgbm_model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("[OK] 模型加载成功")
except FileNotFoundError:
    print("[ERROR] 错误: 请先运行 train_classifier.py 生成模型文件")
    exit(1)

# 2. 转换为ONNX格式
print("\n转换模型为ONNX格式...")

# 定义输入形状: (batch_size, n_features)
# None 表示batch size可变，20是特征数量
initial_type = [('float_input', FloatTensorType([None, 20]))]

try:
    # 转换LightGBM模型
    # 重要：使用 zipmap=False 来获取概率数组而不是字典
    onnx_model = onnxmltools.convert_lightgbm(
        model,
        initial_types=initial_type,
        target_opset=12,  # ONNX opset版本
        zipmap=False  # 关键：输出概率数组而不是类别标签
    )

    # 保存ONNX模型
    onnxmltools.utils.save_model(onnx_model, 'lightgbm_model.onnx')
    print("[OK] 已保存 lightgbm_model.onnx")

except Exception as e:
    print(f"[ERROR] ONNX转换失败: {e}")
    print("\n提示: 请安装依赖:")
    print("  pip install onnxmltools onnxruntime")
    exit(1)

# 3. 保存scaler参数为JSON（C++需要手动实现标准化）
print("\n保存预处理参数...")
scaler_params = {
    'mean': scaler.mean_.tolist(),
    'scale': scaler.scale_.tolist(),
    'n_features': int(scaler.n_features_in_)
}
with open('scaler_params.json', 'w') as f:
    json.dump(scaler_params, f, indent=2)
print("[OK] 已保存 scaler_params.json")

# 4. 保存类别映射
labels = ['DH', 'KD', 'PS10', 'PS10-H', 'QZ', 'YM']
label_mapping = {i: label for i, label in enumerate(labels)}
with open('label_mapping.json', 'w', encoding='utf-8') as f:
    json.dump(label_mapping, f, indent=2, ensure_ascii=False)
print("[OK] 已保存 label_mapping.json")

# 5. 测试ONNX模型
print("\n测试ONNX模型...")
try:
    import onnxruntime as ort

    # 创建推理会话
    sess = ort.InferenceSession('lightgbm_model.onnx')

    # 获取输入输出名称
    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    print(f"  输入名称: {input_name}")
    print(f"  输出名称: {output_name}")
    print(f"  输入形状: {sess.get_inputs()[0].shape}")
    print(f"  输出形状: {sess.get_outputs()[0].shape}")

    # 创建测试数据
    test_data = np.random.randn(1, 20).astype(np.float32)

    # 标准化
    test_data_scaled = (test_data - scaler.mean_) / scaler.scale_

    # 推理
    result = sess.run([output_name], {input_name: test_data_scaled.astype(np.float32)})
    pred_label = np.argmax(result[0], axis=1)[0]

    print(f"\n[OK] ONNX模型测试成功!")
    print(f"  测试预测结果: {labels[pred_label]}")

except ImportError:
    print("  [WARN] 未安装onnxruntime，跳过测试")
    print("  安装命令: pip install onnxruntime")
except Exception as e:
    print(f"  [WARN] 测试失败: {e}")

print("\n" + "="*60)
print("ONNX导出完成！")
print("="*60)
print("\nC++部署所需文件:")
print("  1. lightgbm_model.onnx - ONNX模型文件")
print("  2. scaler_params.json - 特征标准化参数")
print("  3. label_mapping.json - 类别映射表")
print("\n下一步:")
print("  1. 在C++项目中安装 ONNX Runtime")
print("  2. 使用提供的C++代码加载模型")
print("="*60)
