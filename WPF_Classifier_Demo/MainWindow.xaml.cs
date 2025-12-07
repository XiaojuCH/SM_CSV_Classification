using System;
using System.IO;
using System.Windows;
using System.Windows.Media;
using Microsoft.Win32;
using OxyPlot;
using OxyPlot.Axes;
using OxyPlot.Series;

namespace ClassifierDemo
{
    public partial class MainWindow : Window
    {
        private Classifier? classifier;
        private readonly string[] classNames = { "DH", "KD", "PS10", "PS10-H", "QZ", "YM" };
        private readonly OxyColor[] classColors =
        {
            OxyColor.FromRgb(255, 99, 132),   // 红色 - DH
            OxyColor.FromRgb(54, 162, 235),   // 蓝色 - KD
            OxyColor.FromRgb(255, 206, 86),   // 黄色 - PS10
            OxyColor.FromRgb(75, 192, 192),   // 青色 - PS10-H
            OxyColor.FromRgb(153, 102, 255),  // 紫色 - QZ
            OxyColor.FromRgb(255, 159, 64)    // 橙色 - YM
        };

        public MainWindow()
        {
            InitializeComponent();
            InitializeClassifier();
        }

        private void InitializeClassifier()
        {
            try
            {
                classifier = new Classifier();

                // 初始化分类器（DLL 和模型文件应该在程序目录下）
                string baseDir = AppDomain.CurrentDomain.BaseDirectory;
                string modelPath = Path.Combine(baseDir, "lightgbm_model.onnx");
                string scalerPath = Path.Combine(baseDir, "scaler_params.json");
                string labelPath = Path.Combine(baseDir, "label_mapping.json");

                // 检查文件是否存在
                if (!File.Exists(modelPath) || !File.Exists(scalerPath) || !File.Exists(labelPath))
                {
                    StatusText.Text = "状态: ❌ 缺少模型文件";
                    ResultText.Text = "错误: 找不到模型文件！\n\n" +
                                     "请确保以下文件在程序目录下:\n" +
                                     "- ClassifierDLL.dll\n" +
                                     "- onnxruntime.dll\n" +
                                     "- lightgbm_model.onnx\n" +
                                     "- scaler_params.json\n" +
                                     "- label_mapping.json";
                    return;
                }

                classifier.InitializeClassifier(modelPath, scalerPath, labelPath);

                StatusText.Text = "状态: ✓ 分类器已初始化";
                ResultText.Text = "分类器初始化成功！\n\n" +
                                 "请选择功能:\n" +
                                 "- 单样本预测: 输入20个特征值进行预测\n" +
                                 "- 批量 CSV 预测: 从 CSV 文件批量预测";
            }
            catch (Exception ex)
            {
                StatusText.Text = "状态: ❌ 初始化失败";
                ResultText.Text = $"初始化失败:\n{ex.Message}\n\n" +
                                 "请检查:\n" +
                                 "1. DLL 文件是否存在\n" +
                                 "2. 模型文件是否完整\n" +
                                 "3. ONNX Runtime 版本是否正确";
                MessageBox.Show($"初始化失败: {ex.Message}", "错误",
                    MessageBoxButton.OK, MessageBoxImage.Error);
            }
        }

        private void SinglePredictButton_Click(object sender, RoutedEventArgs e)
        {
            if (classifier == null)
            {
                MessageBox.Show("分类器未初始化", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            try
            {
                // 示例数据（你可以修改为从界面输入）
                float[] features = new float[20]
                {
                    581.722f, -0.411162f, -0.262966f, 0.1029f, 355.387f,
                    454.773f, -0.31918f, -0.311743f, 0.148278f, 469.285f,
                    283.031f, -0.330887f, 0.247207f, 0.24066f, 395.569f,
                    434.787f, -0.419953f, -0.0362641f, 0.120564f, 524.24f
                };

                ResultText.Text = "正在预测...\n";

                var result = classifier.Predict(features);

                ResultText.Text = "=== 单样本预测结果 ===\n\n";
                ResultText.Text += $"预测类别: {result.ClassName}\n";
                ResultText.Text += $"置信度: {result.Confidence:P2}\n\n";
                ResultText.Text += "所有类别概率:\n";

                for (int i = 0; i < result.Probabilities.Length; i++)
                {
                    ResultText.Text += $"  {classNames[i]}: {result.Probabilities[i]:P4}\n";
                }

                ResultText.Text += "\n特征值 (前5个):\n";
                for (int i = 0; i < 5; i++)
                {
                    ResultText.Text += $"  特征{i + 1}: {features[i]}\n";
                }

                // 更新图表显示 - 直接使用 DLL 返回的数据
                UpdateCharts(result);
            }
            catch (Exception ex)
            {
                MessageBox.Show($"预测失败: {ex.Message}", "错误",
                    MessageBoxButton.OK, MessageBoxImage.Error);
                ResultText.Text = $"预测失败:\n{ex.Message}";
            }
        }

        private void BatchPredictButton_Click(object sender, RoutedEventArgs e)
        {
            if (classifier == null)
            {
                MessageBox.Show("分类器未初始化", "错误", MessageBoxButton.OK, MessageBoxImage.Error);
                return;
            }

            try
            {
                // 打开文件对话框选择 CSV 文件
                OpenFileDialog openFileDialog = new OpenFileDialog
                {
                    Filter = "CSV 文件 (*.csv)|*.csv|所有文件 (*.*)|*.*",
                    Title = "选择 CSV 文件"
                };

                if (openFileDialog.ShowDialog() == true)
                {
                    ResultText.Text = $"正在处理文件: {Path.GetFileName(openFileDialog.FileName)}\n";
                    ResultText.Text += "请稍候...\n\n";

                    var batchResult = classifier.PredictFromCSV(openFileDialog.FileName);

                    ResultText.Text = "=== 批量预测结果 ===\n\n";
                    ResultText.Text += $"文件: {Path.GetFileName(openFileDialog.FileName)}\n";
                    ResultText.Text += $"总样本数: {batchResult.SampleCount}\n\n";

                    // 显示每个样本的结果
                    for (int i = 0; i < batchResult.SampleCount; i++)
                    {
                        var result = batchResult.Results[i];
                        ResultText.Text += $"[样本 {i + 1}]\n";
                        ResultText.Text += $"  预测: {result.ClassName}\n";
                        ResultText.Text += $"  置信度: {result.Confidence:P2}\n";

                        // 显示概率分布
                        ResultText.Text += "  概率: ";
                        string[] classNames = { "DH", "KD", "PS10", "PS10-H", "QZ", "YM" };
                        for (int j = 0; j < result.Probabilities.Length; j++)
                        {
                            if (result.Probabilities[j] > 0.01) // 只显示大于1%的概率
                            {
                                ResultText.Text += $"{classNames[j]}={result.Probabilities[j]:P1} ";
                            }
                        }
                        ResultText.Text += "\n\n";
                    }

                    ResultText.Text += "=== 预测完成 ===\n";

                    // 显示最后一个样本的图表
                    if (batchResult.SampleCount > 0)
                    {
                        UpdateCharts(batchResult.Results[batchResult.SampleCount - 1]);
                    }

                    MessageBox.Show($"成功处理 {batchResult.SampleCount} 个样本！",
                        "完成", MessageBoxButton.OK, MessageBoxImage.Information);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show($"批量预测失败: {ex.Message}", "错误",
                    MessageBoxButton.OK, MessageBoxImage.Error);
                ResultText.Text = $"批量预测失败:\n{ex.Message}";
            }
        }

        protected override void OnClosed(EventArgs e)
        {
            classifier?.Dispose();
            base.OnClosed(e);
        }

        /// <summary>
        /// 更新图表显示 - 直接使用 DLL 返回的概率数据
        /// </summary>
        private void UpdateCharts(Classifier.PredictionResult result)
        {
            // 绘制柱状图
            UpdateBarChart(result);

            // 绘制饼图
            UpdatePieChart(result);
        }

        /// <summary>
        /// 更新柱状图 - 显示所有类别的概率
        /// </summary>
        private void UpdateBarChart(Classifier.PredictionResult result)
        {
            var plotModel = new PlotModel
            {
                Title = $"预测结果: {result.ClassName} ({result.Confidence:P2})",
                TitleFontSize = 16,
                TitleFontWeight = OxyPlot.FontWeights.Bold
            };

            // 创建柱状图系列
            var barSeries = new BarSeries
            {
                FillColor = OxyColor.FromRgb(54, 162, 235),
                StrokeThickness = 1,
                StrokeColor = OxyColors.White
            };

            // 添加数据 - 直接使用 DLL 返回的 Probabilities 数组
            for (int i = 0; i < result.Probabilities.Length; i++)
            {
                barSeries.Items.Add(new BarItem
                {
                    Value = result.Probabilities[i] * 100,  // 转换为百分比
                    Color = classColors[i]
                });
            }

            plotModel.Series.Add(barSeries);

            // 配置 Y 轴（类别）
            var categoryAxis = new CategoryAxis
            {
                Position = AxisPosition.Left,
                Key = "CategoryAxis",
                ItemsSource = classNames,
                FontSize = 12
            };
            plotModel.Axes.Add(categoryAxis);

            // 配置 X 轴（概率百分比）
            var valueAxis = new LinearAxis
            {
                Position = AxisPosition.Bottom,
                Title = "概率 (%)",
                Minimum = 0,
                Maximum = 100,
                MajorGridlineStyle = LineStyle.Solid,
                MinorGridlineStyle = LineStyle.Dot,
                FontSize = 12
            };
            plotModel.Axes.Add(valueAxis);

            // 更新图表
            BarChartView.Model = plotModel;
        }

        /// <summary>
        /// 更新饼图 - 显示概率分布
        /// </summary>
        private void UpdatePieChart(Classifier.PredictionResult result)
        {
            var plotModel = new PlotModel
            {
                Title = "概率分布",
                TitleFontSize = 16,
                TitleFontWeight = OxyPlot.FontWeights.Bold
            };

            var pieSeries = new PieSeries
            {
                StrokeThickness = 2.0,
                InsideLabelPosition = 0.8,
                AngleSpan = 360,
                StartAngle = 0,
                FontSize = 12
            };

            // 添加数据 - 直接使用 DLL 返回的 Probabilities 数组
            for (int i = 0; i < result.Probabilities.Length; i++)
            {
                double percentage = result.Probabilities[i] * 100;

                // 只显示大于 0.01% 的类别
                if (percentage > 0.01)
                {
                    pieSeries.Slices.Add(new PieSlice(
                        classNames[i],
                        result.Probabilities[i]
                    )
                    {
                        Fill = classColors[i],
                        IsExploded = i == result.ClassIndex  // 突出显示预测的类别
                    });
                }
            }

            plotModel.Series.Add(pieSeries);

            // 更新图表
            PieChartView.Model = plotModel;
        }
    }
}
