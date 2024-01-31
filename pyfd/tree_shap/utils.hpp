// Utility functions and classes for TreeSHAP

#ifndef __UTILS
#define __UTILS

#include <algorithm>
#include <vector>
#include <stack>

using namespace std;

// Custom Types
template <typename T>
using Matrix = vector<vector<T>>;
template <typename T>
using Tensor = vector<vector<vector<T>>>;

template<typename T>
T return_max(T* list, int length){
    T max_element = list[0];
    for (int i(1); i < length; i++){
        if (list[i] > max_element){
            max_element = list[i];
        }
    }
    return max_element;
}


template<typename T>
Matrix<T> createMatrix(int n, int m, T* data){
    Matrix<T> mat;
    for (int i(0); i < n; i++){
        mat.push_back(vector<T> ());
        for (int j(0); j < m; j++){
            mat[i].push_back((data[m * i + j]));
        }
    }
    return mat;
}

template<typename T>
Tensor<T> createTensor(int n, int m, int l, T* data){
    Tensor<T> mat;
    for (int i(0); i < n; i++){
        mat.push_back(Matrix<T> ());
        for (int j(0); j < m; j++){
            mat[i].push_back(vector<T> ());
            for (int k(0); k < l; k++){
                mat[i][j].push_back((data[l * m * i + l * j + k]));
            }
        }
    }
    return mat;
}

template<typename T>
void printMatrix(Matrix<T> mat){
    int n = mat.size();
    int m = mat[0].size();
    for (int i(0); i < n; i++){
        for (int j(0); j < m; j++){
            cout << mat[i][j] << " ";
        }
        cout << "\n";
    }
}

template<typename T>
void printTensor(Tensor<T> mat){
    int n = mat.size();
    int m = mat[0].size();
    int l = mat[0][0].size();
    for (int i(0); i < n; i++){
        for (int j(0); j < m; j++){
            for (int k(0); k < l; k++){
                cout << mat[i][j][k] << " ";
            }
            cout << "\n";
        }
        cout << "\n\n";
    }
}



void compute_W(Matrix<double> &W)
{
    int D = W.size();
    for (double j(0); j < D; j++){
        W[0][j] = 1 / (j + 1);
        W[j][j] = 1 / (j + 1);
    }
    for (double j(2); j < D; j++){
        for (double i(j-1); i > 0; i--){
            W[i][j] = (j - i) / (i + 1) * W[i+1][j];
        }
    }
}

// Return the integer A^B with int A and B
int integer_exp(int base, int expon){
    int res = 1;
    for (int i(0); i < expon; i++){
        res *= base;
    }
    return res;
}


// Assert if s is in S
bool isin_S(int s, int S, vector<int>& branch_I){
    auto it = find(branch_I.begin(), branch_I.end(), s); 

    // If element was found 
    if (it != branch_I.end())  
    { 
        // calculating the index 
        int index = it - branch_I.begin(); 
        if ((S & (1 << index)) > 0) {return true;}
        else {return false;}
    } 
    else { 
        return false;
    } 
}


class FeatureSet {
    // Class that represents the set S_L of features from the root to leaf L
    // As one traverses the decision tree, the features are added and removed 
    // to this set following the root-leaf path. 

    public:
        // default constructor
        FeatureSet(int d, int* I_map);

        // default destructor
        ~FeatureSet() = default;

        // get the cardinality |S_L|
        int size();
        // get the counts for feature i
        int get_feature_count(int i);
        // get the vector of features in S_L in increasing order
        vector<int> get_S_vector();
        // get the vector of features in I(S_L) in increasing order
        vector<int> get_I_S_vector();
        // Add feature to S_L
        void add_feature(int feature);
        // Remove the d first features in the stack
        void remove_features(int d);

    private:
        int d_;
        int n_features_;
        int size_;
        int* I_map_;
        vector<int> S_L_counts_;
        vector<int> I_S_L_counts_;
        stack<int> S_L_stack_;
};

inline FeatureSet::FeatureSet(int d, int* I_map) :
    d_(d),
    n_features_(return_max(I_map, d)+1),
    size_(0),
    I_map_(I_map),
    S_L_counts_(vector<int> (d, 0)),
    I_S_L_counts_(vector<int> (n_features_, 0)),
    S_L_stack_(stack<int> ()) {}


inline int FeatureSet::size() {
    return size_;
}

inline int FeatureSet::get_feature_count(int i) {
    return S_L_counts_[i];
}

inline vector<int> FeatureSet::get_S_vector() {
    vector<int> res (0);
    for (int k(0); k < d_; k++){
        // i is in S_L
        if (S_L_counts_[k] > 0){
            res.push_back(k);
        }
    }
    return res;
}

inline vector<int> FeatureSet::get_I_S_vector() {
    vector<int> res (0);
    for (int k(0); k < n_features_; k++){
        // i is in I(S_L)
        if (I_S_L_counts_[k] > 0){
            res.push_back(k);
        }
    }
    return res;
}

inline void FeatureSet::add_feature(int k) {
    S_L_stack_.push(k);
    if (S_L_counts_[k] == 0){
        size_ += 1;
    }
    S_L_counts_[k] += 1;
    I_S_L_counts_[I_map_[k]] += 1;
}

inline void FeatureSet::remove_features(int n) {
    int last_feature;
    for (int k(0); k < n; k++){
        last_feature = S_L_stack_.top();
        S_L_stack_.pop();
        // Decrease the size of S_L if necessary
        if (S_L_counts_[last_feature] == 1) {
            size_ -= 1;
        }
        S_L_counts_[last_feature] -= 1;
        I_S_L_counts_[I_map_[last_feature]] -= 1;
    }
}




// USEFULL FOR INTERVENTIONAL WITH STACK!!!
// class FeatureSet {
//     // Class that represents the sets S_X and S_Z of features from the root to leaf
//     // As one traverses the decision tree, the features are added and removed 
//     // to these sets following the root-leaf path. 

//     public:
//         // default constructor
//         FeatureSet(int d);

//         // default destructor
//         ~FeatureSet() = default;

//         // cardinality |S_X|
//         int size_SX();
//         // cardinality |S_Z|
//         int size_SZ();
//         // i in SX ?
//         int in_SX(int i);
//         // i in SZ ?
//         int in_SZ(int i);
//         // get the vector of root-leaf features
//         // vector<int> get_feature_path();
//         // Add feature to the path
//         // tag = 0 add nothing to S_X or S_Z
//         // tag = 1 add to S_X
//         // tag = 2 add to S_Z
//         void add_feature(int feature, int tag);
//         // Remove the d first features in the path
//         // This will update SX and SZ accordingly
//         // given the tags
//         void remove_features(int d);
//         bool is_path_empty();

//     private:
//         int d_;
//         int size_SX_;
//         int size_SZ_;
//         vector<int> in_SX_;
//         vector<int> in_SZ_;
//         stack<int> feature_path_;
//         stack<int> tags_;
// };

// inline FeatureSet::FeatureSet(int d) :
//     d_(d),
//     size_SX_(0),
//     size_SZ_(0),
//     in_SX_(vector<int> (d, 0)),
//     in_SZ_(vector<int> (d, 0)),
//     feature_path_(stack<int> ()),
//     tags_(stack<int> ()) {}


// inline int FeatureSet::size_SX() {
//     return size_SX_;
// }

// inline int FeatureSet::size_SZ() {
//     return size_SZ_;
// }

// inline int FeatureSet::in_SX(int i) {
//     return in_SX_[i];
// }

// inline int FeatureSet::in_SZ(int i) {
//     return in_SZ_[i];
// }

// // inline vector<int> FeatureSet::get_feature_vector() {
// //     vector<int> res (0);
// //     for (int i(0); i < d_; i++){
// //         // i is in S_L
// //         if (S_L_counts_[i] > 0){
// //             res.push_back(i);
// //         }
// //     }
// //     return res;
// // }

// inline void FeatureSet::add_feature(int feature, int tag) {
//     feature_path_.push(feature);
//     tags_.push(tag);
//     // Add feature to SX
//     if (tag == 1) {
//         in_SX_[feature] = 1;
//         size_SX_ += 1;
//     }
//     // Add feature to SZ
//     else if (tag == 2) {
//         in_SZ_[feature] = 1;
//         size_SZ_ += 1;
//     }
// }

// inline void FeatureSet::remove_features(int d) {
//     int last_feature, last_tag;
//     for (int i(0); i < d; i++){
//         last_feature = feature_path_.top();
//         last_tag = tags_.top();
//         feature_path_.pop();
//         tags_.pop();
//         // Remove from SX
//         if (last_tag == 1){
//             in_SX_[last_feature] = 0;
//             size_SX_ -= 1;
//         }
//         else if (last_tag == 2){
//             in_SZ_[last_feature] = 0;
//             size_SZ_ -= 1;
//         }
//     }
// }

// inline bool FeatureSet::is_path_empty() {
//     return feature_path_.empty();
// }

#endif