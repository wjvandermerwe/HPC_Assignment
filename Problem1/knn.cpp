#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <string>
#include <utility>
#include <omp.h>

using namespace std;

using FeatureMap = vector<vector<float>>;
using ClusteredOutput = vector<float>;

// Reads a flattened (NUM_SAMPLES x FEATURE_DIM) binary file of floats
FeatureMap read_features(const string &filename, size_t num_samples, size_t feature_dim) {
    size_t total_elements = num_samples * feature_dim;
    vector<float> flat(total_elements);
    
    ifstream file(filename, ios::binary);
    if (!file) {
        throw runtime_error("Could not open file: " + filename);
    }
    
    file.read(reinterpret_cast<char*>(flat.data()), total_elements * sizeof(float));
    if (!file)
        throw runtime_error("Error reading file: " + filename);
    file.close();
    
    // Reshape the flat vector into a vector of vectors.
    FeatureMap features;
    features.reserve(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        vector<float> sample(flat.begin() + i * feature_dim, flat.begin() + (i + 1) * feature_dim);
        features.push_back(move(sample));
    }
    
    return features;
}

// Reads a binary file containing labels.
vector<int> read_labels(const string &filename, size_t num_samples) {
    vector<int> labels(num_samples);
    
    ifstream file(filename, ios::binary);
    if (!file) {
        throw runtime_error("Could not open file: " + filename);
    }
    
    file.read(reinterpret_cast<char*>(labels.data()), num_samples * sizeof(int));
    if (!file)
        throw runtime_error("Error reading file: " + filename);
    file.close();
    
    return labels;
}

// Euclidean distance
float compute_distance() {
    return
}

FeatureMap quick_sort(FeatureMap features, size_t num_samples, size_t feature_dim) {

}



int main() {

    constexpr size_t NUM_SAMPLES = 50000;
    constexpr size_t FEATURE_DIM = 512;
    
    vector<vector<float>> features = read_features("train/train_features.bin", NUM_SAMPLES, FEATURE_DIM);
    vector<int> labels = read_labels("train/train_labels.bin", NUM_SAMPLES);


    
    return 0;
}

