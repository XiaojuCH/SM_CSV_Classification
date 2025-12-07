/**
 * Simple LightGBM ONNX Classifier
 *
 * Function: Load ONNX model and perform prediction
 *
 * Compile:
 * 1. Download ONNX Runtime: https://github.com/microsoft/onnxruntime/releases
 * 2. Extract to directory, e.g. C:\onnxruntime
 * 3. Compile command (Windows):
 *    cl /EHsc /std:c++17 simple_classifier.cpp /I"C:\onnxruntime\include" /link "C:\onnxruntime\lib\onnxruntime.lib"
 * 4. Copy onnxruntime.dll to executable directory
 *
 * Usage:
 *    simple_classifier.exe
 */

#include <onnxruntime_cxx_api.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>

using namespace std;

// Simple JSON array parser
vector<double> parse_json_array(const string& json_content, const string& key) {
    vector<double> result;

    // Find key position
    size_t key_pos = json_content.find("\"" + key + "\"");
    if (key_pos == string::npos) return result;

    // Find array start and end
    size_t array_start = json_content.find("[", key_pos);
    size_t array_end = json_content.find("]", array_start);

    // Extract array content
    string array_str = json_content.substr(array_start + 1, array_end - array_start - 1);

    // Parse numbers
    stringstream ss(array_str);
    string item;
    while (getline(ss, item, ',')) {
        // Remove whitespace
        item.erase(remove_if(item.begin(), item.end(), ::isspace), item.end());
        if (!item.empty()) {
            result.push_back(stod(item));
        }
    }

    return result;
}

// Load label mapping
vector<string> load_labels(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Cannot open file: " << filename << endl;
        return {};
    }

    string content((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());

    vector<string> labels(6);
    for (int i = 0; i < 6; i++) {
        string key = "\"" + to_string(i) + "\"";
        size_t pos = content.find(key);
        if (pos != string::npos) {
            size_t value_start = content.find("\"", pos + key.length()) + 1;
            size_t value_end = content.find("\"", value_start);
            labels[i] = content.substr(value_start, value_end - value_start);
        }
    }

    return labels;
}

// Apply softmax to convert logits to probabilities
vector<float> softmax(const float* logits, int num_classes) {
    vector<float> probs(num_classes);

    // Find max for numerical stability
    float max_logit = logits[0];
    for (int i = 1; i < num_classes; i++) {
        if (logits[i] > max_logit) {
            max_logit = logits[i];
        }
    }

    // Compute exp(logit - max_logit) and sum
    float sum_exp = 0.0f;
    for (int i = 0; i < num_classes; i++) {
        probs[i] = exp(logits[i] - max_logit);
        sum_exp += probs[i];
    }

    // Normalize to get probabilities
    for (int i = 0; i < num_classes; i++) {
        probs[i] /= sum_exp;
    }

    return probs;
}

// Read CSV file and return all rows
// Each row should have exactly 20 features
vector<vector<float>> read_csv(const string& filename) {
    vector<vector<float>> data;

    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Cannot open " << filename << endl;
        return data;
    }

    string line;
    int line_number = 0;

    while (getline(file, line)) {
        line_number++;

        // Skip empty lines
        if (line.empty()) {
            continue;
        }

        vector<float> row;
        stringstream ss(line);
        string value;

        // Split by comma
        while (getline(ss, value, ',')) {
            try {
                row.push_back(stof(value));  // String to float
            } catch (const exception& e) {
                cerr << "Warning: Invalid value at line " << line_number << endl;
                break;
            }
        }

        // Check if row has exactly 20 features
        if (row.size() == 20) {
            data.push_back(row);
        } else if (row.size() > 0) {
            cerr << "Warning: Line " << line_number << " has " << row.size()
                 << " features (expected 20), skipping..." << endl;
        }
    }

    return data;
}

int main(int argc, char* argv[]) {
    cout << "============================================================" << endl;
    cout << "LightGBM ONNX Classifier - Batch Prediction" << endl;
    cout << "============================================================" << endl;

    // Check command line arguments
    if (argc < 2) {
        cout << "\nUsage: " << argv[0] << " <csv_file>" << endl;
        cout << "\nExample:" << endl;
        cout << "  " << argv[0] << " test_data.csv" << endl;
        cout << "  " << argv[0] << " ../DH.csv" << endl;
        cout << "\nCSV file format:" << endl;
        cout << "  - Each line contains exactly 20 features" << endl;
        cout << "  - Features separated by commas" << endl;
        cout << "  - No header row" << endl;
        return 1;
    }

    string csv_filename = argv[1];
    cout << "\nInput CSV file: " << csv_filename << endl;

    try {
        // 1. Load scaler parameters
        cout << "\n[1/4] Loading scaler parameters..." << endl;
        ifstream scaler_file("scaler_params.json");
        if (!scaler_file.is_open()) {
            cerr << "Error: Cannot open scaler_params.json" << endl;
            return 1;
        }

        string scaler_content((istreambuf_iterator<char>(scaler_file)), istreambuf_iterator<char>());
        vector<double> mean = parse_json_array(scaler_content, "mean");
        vector<double> scale = parse_json_array(scaler_content, "scale");

        cout << "  Number of features: " << mean.size() << endl;

        // 2. Load label mapping
        cout << "\n[2/4] Loading label mapping..." << endl;
        vector<string> labels = load_labels("label_mapping.json");
        cout << "  Number of classes: " << labels.size() << endl;
        cout << "  Classes: ";
        for (const auto& label : labels) {
            cout << label << " ";
        }
        cout << endl;

        // 3. Load ONNX model
        cout << "\n[3/4] Loading ONNX model..." << endl;

        // Create environment with minimal logging for faster startup
        cout << "  [Step 1/3] Creating ONNX environment..." << endl;
        Ort::Env env(ORT_LOGGING_LEVEL_ERROR, "LightGBMClassifier");

        // Configure session options for faster loading
        cout << "  [Step 2/3] Configuring session options..." << endl;
        Ort::SessionOptions session_options;

        // CRITICAL: Disable ALL graph optimizations to avoid slow loading
        // Graph optimization can take several minutes for large models
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);

        // Use single thread for inference (simpler and faster to initialize)
        session_options.SetIntraOpNumThreads(1);

        // Load the model
        cout << "  [Step 3/3] Loading model file..." << endl;
        #ifdef _WIN32
        wstring model_path = L"lightgbm_model.onnx";
        Ort::Session session(env, model_path.c_str(), session_options);
        #else
        Ort::Session session(env, "lightgbm_model.onnx", session_options);
        #endif

        cout << "  Model loaded successfully!" << endl;

        // Get input/output info
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name_ptr = session.GetInputNameAllocated(0, allocator);

        // ONNX model has 2 outputs: [0]=label (string), [1]=probabilities (float array)
        // We need the second output (index 1) for probabilities
        auto output_name_ptr = session.GetOutputNameAllocated(1, allocator);

        const char* input_name = input_name_ptr.get();
        const char* output_name = output_name_ptr.get();

        cout << "  Input name: " << input_name << endl;
        cout << "  Output name (probabilities): " << output_name << endl;

        // 4. Read CSV data
        cout << "\n[4/5] Reading CSV data..." << endl;
        vector<vector<float>> all_samples = read_csv(csv_filename);

        if (all_samples.empty()) {
            cerr << "Error: No valid data found in CSV file" << endl;
            return 1;
        }

        cout << "  Total samples: " << all_samples.size() << endl;

        // 5. Perform batch prediction
        cout << "\n[5/5] Starting batch prediction..." << endl;
        cout << "============================================================" << endl;

        // Prepare for batch inference
        const char* input_names[] = {input_name};
        const char* output_names[] = {output_name};
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        // Predict each sample
        for (size_t sample_idx = 0; sample_idx < all_samples.size(); sample_idx++) {
            const vector<float>& features = all_samples[sample_idx];

            // Standardize features
            vector<float> scaled_features(20);
            for (int i = 0; i < 20; i++) {
                scaled_features[i] = (features[i] - mean[i]) / scale[i];
            }

            // Create input tensor
            vector<int64_t> input_shape = {1, 20};
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info,
                scaled_features.data(),
                scaled_features.size(),
                input_shape.data(),
                input_shape.size()
            );

            // Run inference
            auto output_tensors = session.Run(
                Ort::RunOptions{nullptr},
                input_names,
                &input_tensor,
                1,
                output_names,
                1
            );

            // Get output (probabilities from second output)
            // Note: We requested output index 1, so output_tensors[0] contains the probabilities
            float* output_data = output_tensors[0].GetTensorMutableData<float>();

            // Print result for this sample
            cout << "\n[Sample " << (sample_idx + 1) << "]" << endl;
            cout << "  Features (first 5): ";
            for (int i = 0; i < 5; i++) {
                cout << features[i] << " ";
            }
            cout << "..." << endl;

            // The output is already probabilities (not logits), no need for softmax
            // Find class with maximum probability
            int max_idx = 0;
            float max_prob = output_data[0];
            for (int i = 1; i < 6; i++) {
                if (output_data[i] > max_prob) {
                    max_prob = output_data[i];
                    max_idx = i;
                }
            }

            cout << "  Probabilities: ";
            for (int i = 0; i < 6; i++) {
                cout << labels[i] << "=" << (output_data[i] * 100) << "% ";
            }
            cout << endl;

            cout << "  Prediction: " << labels[max_idx]
                 << " (Confidence: " << (max_prob * 100) << "%)" << endl;
        }

        cout << "\n============================================================" << endl;
        cout << "Batch prediction completed!" << endl;
        cout << "Total samples processed: " << all_samples.size() << endl;
        cout << "============================================================" << endl;

    } catch (const Ort::Exception& e) {
        cerr << "\nONNX Runtime error: " << e.what() << endl;
        return 1;
    } catch (const exception& e) {
        cerr << "\nError: " << e.what() << endl;
        return 1;
    }

    return 0;
}
