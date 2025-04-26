#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <string>
#include <utility>
#include <unordered_map>
#include <chrono>
#include <iomanip>
#include <ctime>

using namespace std;

using Feature = vector<float>;
using Features = vector<Feature>;
using FeaturePairs = vector<pair<float, int>>;
using TimerType =  chrono::time_point<chrono::steady_clock>;

struct Metric {
    std::string name;
    double      accuracy;
    double      total;
    double      dist;
    double      sort;
};

void print_metrics_table(const vector<Metric>& M) {
    if (M.empty()) return;
    double serial_total = M[0].total;

    // header
    std::cout << "| Implementation | Accuracy (%) | Total Time (s) | Speedup vs Serial | "
                 "Distance Time (s) | Distance % | Sorting Time (s) | Sorting % |\n"
              << "|----------------|:------------:|:--------------:|:-----------------:|"
                 ":-----------------:|:----------:|:----------------:|:---------:|\n";

    // rows
    for (auto const& m : M) {
        double speedup      = serial_total / m.total;
        double dist_pct     = 100.0 * m.dist  / m.total;
        double sorting_pct  = 100.0 * m.sort  / m.total;

        std::cout
            << "| " << std::left << std::setw(14) << m.name << " | "
            << std::right << std::fixed << std::setprecision(2)
            << std::setw(12) << m.accuracy    << " | "
            << std::setw(14) << m.total       << " | "
            << std::setw(17) << speedup       << "× | "
            << std::setw(17) << m.dist        << " | "
            << std::setw(8 ) << dist_pct      << "% | "
            << std::setw(16) << m.sort        << " | "
            << std::setw(8 ) << sorting_pct   << "% |\n";
    }
}

Features read_features(const std::string &filename,
                         size_t feature_dim)
{
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Could not open file: " + filename);
    size_t bytes = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t total_floats = bytes / sizeof(float);
    size_t num_samples = total_floats / feature_dim;

    // read all floats at once
    std::vector<float> flat(total_floats);
    file.read(reinterpret_cast<char*>(flat.data()), bytes);
    if (!file) throw std::runtime_error("Error reading file: " + filename);

    // reshape into num_samples × feature_dim
    Features features;
    features.reserve(num_samples);
    for (size_t i = 0; i < num_samples; ++i) {
        features.emplace_back(
            flat.begin() + i*feature_dim,
            flat.begin() + (i+1)*feature_dim
        );
    }
    return features;
}

vector<int> read_labels(const string &filename) {
    ifstream file(filename, ios::binary | ios::ate);
    if (!file) {
        throw runtime_error("Could not open file: " + filename);
    }
    size_t bytes = file.tellg();
    size_t num_samples = bytes / sizeof(int);
    // rewind
    file.seekg(0, ios::beg);

    vector<int> labels(num_samples);
    file.read(reinterpret_cast<char*>(labels.data()), bytes);
    if (!file)
        throw runtime_error("Error reading file: " + filename);

    file.close();
    
    return labels;
}

TimerType main_timer_start(string name) {
    auto t0 = std::chrono::high_resolution_clock::now();
    auto now = std::chrono::system_clock::now();

    std::time_t now_tt = std::chrono::system_clock::to_time_t(now);
    std::cout << name << "Starting at: "
              << std::put_time(std::localtime(&now_tt),
                               "%Y-%m-%d %H:%M:%S")
              << '\n';

    return t0;
}

double timer_end(TimerType start, string name) {
    auto t1 = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration<double>(t1 - start).count();
    std::cout << "Elapsed: "
              << diff
              << " seconds\n";
    auto now = std::chrono::system_clock::now();
    auto now_tt = std::chrono::system_clock::to_time_t(now);
    std::cout << name << "Ended at: "
              << std::put_time(std::localtime(&now_tt),
                               "%Y-%m-%d %H:%M:%S")
              << '\n';
    return diff;
}

float compute_distance(const Feature &from_feature, const Feature &to_feature) {
    float sum = 0.0f;
    size_t from_feature_dim = from_feature.size();
    for (size_t i = 0; i < from_feature_dim; ++i) {
        float diff = to_feature[i] - from_feature[i];
        sum += diff * diff;
    }
    return sqrt(sum);
}

size_t lomuto_partition(FeaturePairs &arr, size_t lo, size_t hi) {
    float pivot = arr[hi].first;
    size_t i = lo;
    for (size_t j = lo; j < hi; ++j) {
        if (arr[j].first < pivot) {
            std::swap(arr[i], arr[j]);
            ++i;
        }
    }
    std::swap(arr[i], arr[hi]);
    return i;
}

size_t hoare_partition(std::vector<float> &arr, int lo, int hi) {
    float pivot = arr[lo + (hi - lo) / 2];
    int i = lo - 1;
    int j = hi + 1;
    while (true) {
        // move i right until arr[i] ≥ pivot
        do { ++i; } while (arr[i] < pivot);
        // move j left until arr[j] ≤ pivot
        do { --j; } while (arr[j] > pivot);
        if (i >= j)
            return j;
        std::swap(arr[i], arr[j]);
    }
}

void quick_sort(FeaturePairs &distances, const size_t lo, const size_t hi) {
    if (lo < hi) {
        size_t p = lomuto_partition(distances, lo, hi);
        if (p > 0)
            quick_sort(distances, lo, p - 1);
        quick_sort(distances, p + 1, hi);
    }
}

FeaturePairs compute_distances(const size_t i, const Features &test_features,
    const vector<int> &train_labels,
    const Features &train_features, double &duration) {

    FeaturePairs dist_lbl;
    dist_lbl.reserve(train_features.size());

    auto s0 = std::chrono::high_resolution_clock::now();
    for (size_t j = 0; j < train_features.size(); ++j) {
        float d = compute_distance(train_features[j], test_features[i]);
        dist_lbl.emplace_back(d, train_labels[j]);
    }
    auto s1 = std::chrono::high_resolution_clock::now();
    duration += std::chrono::duration<double>(s1 - s0).count();

    return dist_lbl;
}

void seq_sort(FeaturePairs distances, double &duration) {
    auto s0 = std::chrono::high_resolution_clock::now();
    quick_sort(distances, 0, distances.size() - 1);
    auto s1 = std::chrono::high_resolution_clock::now();
    duration += std::chrono::duration<double>(s1 - s0).count();
}

void par_sec_sort(FeaturePairs &distances, double &duration) {
    auto s0 = std::chrono::high_resolution_clock::now();

    size_t lo = 0, hi = distances.size() - 1;
    if (lo < hi) {
        size_t p = lomuto_partition(distances, lo, hi);

#pragma omp parallel sections
        {
#pragma omp section
            {
                if (p > lo)
                    quick_sort(distances, lo, p - 1);
            }
#pragma omp section
{
    if (p + 1 < hi)
        quick_sort(distances, p + 1, hi);
}
        }
    }

    auto s1 = std::chrono::high_resolution_clock::now();
    duration += std::chrono::duration<double>(s1 - s0).count();
}

void par_quick_sort(FeaturePairs &A, int lo, int hi) {
    if (lo < hi) {
        int p = lomuto_partition(A, lo, hi);
#pragma omp task shared(A) firstprivate(lo, p)
        par_quick_sort(A, lo, p - 1);
#pragma omp task shared(A) firstprivate(p, hi)
        par_quick_sort(A, p + 1, hi);
#pragma omp taskwait
    }
}

void par_task_sort(FeaturePairs &distances, double &duration) {
    auto s0 = std::chrono::high_resolution_clock::now();

#pragma omp parallel
#pragma omp single nowait
    par_quick_sort(distances, 0, distances.size() - 1);

    auto s1 = std::chrono::high_resolution_clock::now();
    duration += std::chrono::duration<double>(s1 - s0).count();
}


struct Settings {
    bool use_seq_sort;
    bool use_par_sec_sort;
    bool use_par_task_sort;
    bool use_seq_dist;
    bool user_par_dist;
    size_t testing_num;
};

double calculate_accuracy(const vector<int> &preds, const vector<int> &labels) {
    double correct = 0.0;
    for (size_t i = 0; i < preds.size(); ++i) {
        if (preds[i] == labels[i])
            ++correct;
    }
    return correct / preds.size();
}

void vote_and_record_predictions(const size_t K, FeaturePairs &distances, vector<int> preds, const size_t i) {
    unordered_map<int,int> count;
    for (size_t k = 0; k < K; ++k) {
        int index = distances[k].second;
        ++count[index];
    }

    preds[i] = std::max_element(
        count.begin(), count.end(), [](auto &a, auto &b){ return a.second < b.second; }
    )->first;
}

Metric knn_predict(
    Settings settings,
    vector<int> &p_test_labels,
    Features &p_test_features,
    vector<int> &p_train_labels,
    Features &p_train_features,
    size_t K,
    string name)
{
    size_t n = std::min<size_t>(settings.testing_num, p_test_features.size());
    Features test_features {
        p_test_features.begin(),
        p_test_features.begin() + n
    };
    vector<int> test_labels {
        p_test_labels.begin(),
        p_test_labels.begin() + n
    };

    Features train_features {
        p_train_features.begin() + n,
        p_train_features.end()
    };
    vector<int> train_labels {
        p_train_labels.begin() + n,
        p_train_labels.end()
    };

    Metric m;
    m.name = name;
    vector<int> preds(test_features.size());
    auto start = main_timer_start(name);

    if (settings.use_seq_dist) {
        for (size_t i = 0; i < test_features.size(); ++i) {
            auto distances = compute_distances(
                i, test_features,
                train_labels, train_features,
                m.dist
            );

            if (settings.use_par_sec_sort) {
                par_sec_sort(distances, m.sort);
            } else if (settings.use_par_task_sort) {
                par_task_sort(distances, m.sort);
            } else {
                seq_sort(distances, m.sort);
            }

            vote_and_record_predictions(K, distances, preds, i);
        }
    } else {
#pragma omp parallel for schedule(dynamic) reduction(+:m.dist,m.sort)
        for (size_t i = 0; i < test_features.size(); ++i) {
            auto distances = compute_distances(
                i, test_features,
                train_labels, train_features,
                m.dist
            );

            if (settings.use_par_sec_sort) {
                par_sec_sort(distances, m.sort);
            } else if (settings.use_par_task_sort) {
                par_task_sort(distances, m.sort);
            } else {
                seq_sort(distances, m.sort);
            }

            vote_and_record_predictions(K, distances, preds, i);
        }
    }
    auto end = timer_end(start, name);
    m.total = end;
    m.accuracy = calculate_accuracy(preds, test_labels);
    return m;
}

int main() {
    constexpr size_t FEATURE_DIM = 512;
    
    Features train_features = read_features("../Problem1/data/train/train_features.bin", FEATURE_DIM);
    Features test_features = read_features("../Problem1/data/test/test_features.bin", FEATURE_DIM);
    vector<int> train_labels = read_labels("../Problem1/data/train/train_labels.bin");
    vector<int> test_labels = read_labels("../Problem1/data/test/test_labels.bin");

    vector<Metric> metrics(6);
    size_t thres = 1000;
    metrics[0] = knn_predict(Settings{true, false, false, true, false,thres}, test_labels, test_features, train_labels, train_features, 5, "Only Sequential");
    metrics[1] = knn_predict(Settings{false, true, false, true, false,thres}, test_labels, test_features, train_labels, train_features, 5, "Sequential Distances, Parallel Sections Sort");
    metrics[2] = knn_predict(Settings{false, false, true, true, false,thres}, test_labels, test_features, train_labels, train_features, 5, "Sequential Distances, Parallel Tasks Sort");
    metrics[3] = knn_predict(Settings{true, false, false, false, true,thres}, test_labels, test_features, train_labels, train_features, 5, "Parallel Distances, Sequential Sort");
    metrics[4] = knn_predict(Settings{false, true, true, false, true,thres}, test_labels, test_features, train_labels, train_features, 5, "Parallel Distances, Parallel Tasks Sort");
    metrics[5] = knn_predict(Settings{false, true, false, false, true, thres}, test_labels, test_features, train_labels, train_features, 5, "Parallel Distances, Parallel Sections Sort");

    print_metrics_table(metrics);
    return 0;

}

