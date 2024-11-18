#include <string.h>
#include <stdexcept>
#include "recursive_treeshap.hpp"
// #include "stack_treeshap.hpp"
#include "leaf_treeshap.hpp"
#include "waterfall_treeshap.hpp"
#include "utils.hpp"


////// Wrapping the C++ functions with a C interface //////


// ********************************************************************************
//                                   Parameters
// ********************************************************************************
//     int Nx : number of foreground instances
//     int Nz : number of background instances
//     int Nt : number of trees
//     int d  : number of input columns
//     int depth : max depth of the tree ensemble
//     double* foreground : row major (Nx, d) array of foreground data
//     double* background : row major (Nz, d) array of background data
//     int* I_map : function mapping features to their feature group
//     double* threshold : row major (Nt, depth) array of thresholds
//     double* value_: row major (Nt, depth) array of values
//     int* feature : row major (Nt, depth) array of feature index
//     int* left_child : row major (Nt, depth) array of left child index
//     int* right_child : row major (Nt, depth) array of right child index
//     int anchored : value 1 for anchored decomposition and 0 for interventional
//     int sym : value 1 if foreground is the same as background
//     double* result : row major (Nx, Nz, n_features) if anchored else (Nx, n_features)
// ********************************************************************************
extern "C"
int main_recurse_treeshap(int Nx, int Nz, int Nt, int d, int depth, double* foreground, double* background,
                            int* I_map, double* threshold, double* value, int* feature, 
                            int* left_child, int* right_child, int anchored, int sym, double* result) {
    int f_index, b_index;
    // Structure to store tree data
    struct TreeEnsemble tree;
    //Structure to encode the mapping I
    struct Image image;
    init_image(&image, d, I_map);
    // The current shapley values
    double* phi = (double*) calloc(image.n_features, sizeof(double));
    // Precompute the SHAP weights
    Matrix<double> W(image.n_features, vector<double> (image.n_features));
    compute_W(W);
    
    // Main loop
    progressbar bar(Nx);
    // Iterate over all foreground instances
    for (int i(0); i < Nx; i++){
        f_index = i * d;
        init_tree(&tree, depth, threshold, value, feature, left_child, right_child);
        // Iterate over all trees
        for (int t(0); t < Nt; t++){
            // Iterate over all background instances
            for (int j = sym ? i+1 : 0; j < Nz; j++){
                b_index = j * d;
                // Initialize shapley values at zero
                memset(phi, 0, image.n_features * sizeof(double));
                
                // Start the recursion
                recurse_treeshap(0, &foreground[f_index], &background[b_index], &tree, &image, W, phi);
                
                // Store the results
                if (anchored){
                    for (int f(0); f < image.n_features; f++){
                        result[Nz*image.n_features*i + image.n_features*j + f] += phi[f];
                        if (sym){
                            result[Nz*image.n_features*j + image.n_features*i + f] -= phi[f];
                        }
                    }
                }
                else {
                    for (int f(0); f < image.n_features; f++){
                        result[image.n_features*i + f] += phi[f];
                        if (sym){
                            result[image.n_features*j + f] -= phi[f];
                        }
                    }
                }
            }
            next_tree(&tree);
        }
        bar.update();
    }
    if (!anchored){
        // Rescale w.r.t the number of background instances
        for (int i(0); i < Nx*image.n_features; i++){
            result[i] /= Nz;
        }
    }
    std::cout << std::endl;
    free_image(&image);
    free(phi);

    return 0;
}





// ********************************************************************************
//                                   Parameters
// ********************************************************************************
//     int Nx : number of foreground instances
//     int Nz : number of background instances
//     int Nt : number of trees
//     int d  : number of input features
//     int depth : max depth of the tree ensemble
//     double* foreground : row major (Nx, d) array of foreground data
//     double* background : row major (Nz, d) array of background data
//     int* I_map : function mapping features to their feature group
//     double* threshold : row major (Nt, depth) array of thresholds
//     double* value_: row major (Nt, depth) array of values
//     int* feature : row major (Nt, depth) array of feature index
//     int* left_child : row major (Nt, depth) array of left child index
//     int* right_child : row major (Nt, depth) array of right child index
//     int sym : value 1 if foreground is the same as background
//     double* result : row major (Nx, Nz, n_features)
// ********************************************************************************
extern "C"
int main_recurse_additive(int Nx, int Nz, int Nt, int d, int depth, double* foreground, double* background,
                        int* I_map, double* threshold, double* value, int* feature, 
                        int* left_child, int* right_child, int sym, double* result) {
    // Cast to a boolean
    int f_index, b_index, result_index_1, result_index_2;
    // Structure to store tree data
    struct TreeEnsemble tree;
    //Structure to encode the mapping I
    struct Image image;
    init_image(&image, d, I_map);
    
    // Main loop
    progressbar bar(Nx);
    // Iterate over all foreground instances
    for (int i(0); i < Nx; i++){
        f_index = i * d;
        init_tree(&tree, depth, threshold, value, feature, left_child, right_child);
        // Iterate over all trees
        for (int t(0); t < Nt; t++){
            // Iterate over all background instances
            for (int j = sym ? i+1 : 0; j < Nz; j++){
                b_index = j * d;
                if (sym){
                    vector<int> ISX, ISZ;
                    result_index_1 = Nz*image.n_features*i + image.n_features*j;
                    result_index_2 = Nz*image.n_features*j + image.n_features*i;
                    // Start the recursion
                    recurse_additive_sym(0, &foreground[f_index], &background[b_index], &tree, &image, 
                                            ISX, ISZ, &result[result_index_1], &result[result_index_2]);
                }
                else {
                    result_index_1 = Nz*image.n_features*i + image.n_features*j;
                    // Start the recursion
                    recurse_additive(0, &foreground[f_index], &background[b_index], &tree, &image, &result[result_index_1]);
                }
            }
            next_tree(&tree);
        }
        bar.update();
    }
    std::cout << std::endl;
    free_image(&image);
    
    return 0;
}



// ********************************************************************************
//                                   Parameters
// ********************************************************************************
//     int Nx : number of foreground instances
//     int Nz : number of background instances
//     int Nt : number of trees
//     int d  : number of input features
//     int depth : max depth of the tree ensemble
//     double* foreground : row major (Nx, d) array of foreground data
//     double* background : row major (Nz, d) array of background data
//     int* I_map : function mapping features to their feature group
//     double* threshold : row major (Nt, depth) array of thresholds
//     double* value_: row major (Nt, depth) array of values
//     int* feature : row major (Nt, depth) array of feature index
//     int* left_child : row major (Nt, depth) array of left child index
//     int* right_child : row major (Nt, depth) array of right child index
//     double* result : row major (Nx, n_features, n_features)
// ********************************************************************************
extern "C"
int main_taylor_treeshap(int Nx, int Nz, int Nt, int d, int depth, double* foreground, double* background,
                            int* I_map, double* threshold, double* value, int* feature, 
                            int* left_child, int* right_child, double* result) {
    // Cast to a boolean
    int f_index, b_index, result_index;
    // Tree structure
    struct TreeEnsemble tree;
    //Structure to encode the mapping I
    struct Image image;
    init_image(&image, d, I_map);
    // Precompute the SHAP weights
    Matrix<double> W(image.n_features, vector<double> (image.n_features));
    compute_W(W);
    
    // Main loop
    progressbar bar(Nx);
    // Iterate over all foreground instances
    for (int i(0); i < Nx; i++){
        f_index = i * d;
        result_index = image.n_features*image.n_features*i;
        init_tree(&tree, depth, threshold, value, feature, left_child, right_child);
        // Iterate over all trees
        for (int t(0); t < Nt; t++){
            // Iterate over all background instances
            for (int j(0); j < Nz; j++){
                b_index = j * d;
                // Set I(S_X) U I(S_Z)
                vector<int> ISX_U_ISZ;

                // Start the recursion
                recurse_taylor_treeshap(0, &foreground[f_index], &background[b_index], &tree, &image, 
                                                                ISX_U_ISZ, W, &result[result_index]);
            }
            next_tree(&tree);
        }
        bar.update();
    }
    std::cout << std::endl;
    free_image(&image);

    return 0;
}





// ********************************************************************************
//                                   Parameters
// ********************************************************************************
//     int Nx : number of foreground instances
//     int Nz : number of background instances
//     int Nt : number of trees
//     int d  : number of input features
//     int depth : max depth of the tree ensemble
//     int M : TODO
//     int max_var : max(n_features, depth) maximum number of players in leaf game
//     double* foreground : row major (Nx, d) array of foreground data
//     double* background : row major (Nz, d) array of background data
//     int* I_map : function mapping features to their feature group
//     double* value_: row major (Nt, depth) array of values
//     int* feature : row major (Nt, depth) array of feature index
//     int* left_child : row major (Nt, depth) array of left child index
//     int* right_child : row major (Nt, depth) array of right child index
//     double* partition_min_ : TODO
//     double* partition_max_ : TODO
//     double* result : row major (Nx, n_features)
// ********************************************************************************
extern "C"
int main_leaf_treeshap(int Nx, int Nz, int Nt, int d, int depth, int M, int max_var,
                         double* foreground, double* background, int* I_map,
                         double* value_, int* feature_, int* left_child_, int* right_child_,
                         double* partition_min_, double* partition_max_, double* result) {
    
    // Load data instances
    Matrix<double> X_f = createMatrix<double>(Nx, d, foreground);
    Matrix<double> X_b = createMatrix<double>(Nz, d, background);

    // Load tree structure
    Matrix<double> value = createMatrix<double>(Nt, depth, value_);
    Matrix<int> feature = createMatrix<int>(Nt, depth, feature_);
    Matrix<int> left_child  = createMatrix<int>(Nt, depth, left_child_);
    Matrix<int> right_child = createMatrix<int>(Nt, depth, right_child_);
    Tensor<double> partition_min = createTensor<double>(Nt, M, d, partition_min_);
    Tensor<double> partition_max = createTensor<double>(Nt, M, d, partition_max_);
    
    leaf_treeshap(X_f, X_b, feature, left_child, right_child, partition_min, partition_max, value, I_map, max_var, result);
    cout << endl;

    return 0;
}



// ********************************************************************************
//                                   Parameters
// ********************************************************************************
//     int Nx : number of foreground instances
//     int Nz : number of background instances
//     int Nt : number of trees
//     int d  : number of input features
//     int depth : max depth of the tree ensemble
//     int M : TODO
//     double* foreground : row major (Nx, d) array of foreground data
//     double* background : row major (Nz, d) array of background data
//     int* I_map : function mapping features to their feature group
//     double* value_: row major (Nt, depth) array of values
//     int* feature : row major (Nt, depth) array of feature index
//     int* left_child : row major (Nt, depth) array of left child index
//     int* right_child : row major (Nt, depth) array of right child index
//     double* partition_min_ : TODO
//     double* partition_max_ : TODO
//     double* result : row major (Nx, n_features)
// ********************************************************************************
extern "C"
int main_leaf_additive(int Nx, int Nz, int Nt, int d, int depth, int M,
                         double* foreground, double* background, int* I_map,
                         double* value_, int* feature_, int* left_child_, int* right_child_,
                         double* partition_min_, double* partition_max_, double* result) {
    
    // Load data instances
    Matrix<double> X_f = createMatrix<double>(Nx, d, foreground);
    Matrix<double> X_b = createMatrix<double>(Nz, d, background);

    // Load tree structure
    Matrix<double> value = createMatrix<double>(Nt, depth, value_);
    Matrix<int> feature = createMatrix<int>(Nt, depth, feature_);
    Matrix<int> left_child  = createMatrix<int>(Nt, depth, left_child_);
    Matrix<int> right_child = createMatrix<int>(Nt, depth, right_child_);
    Tensor<double> partition_min = createTensor<double>(Nt, M, d, partition_min_);
    Tensor<double> partition_max = createTensor<double>(Nt, M, d, partition_max_);
    
    leaf_additive(X_f, X_b, feature, left_child, right_child, partition_min, partition_max, value, I_map, result);
    cout << endl;

    return 0;
}



// extern "C"
// int main_observ_treeshap(int Nx, int Nz, int Nt, int d, int depth, int M, int max_var, 
//                          double* foreground, double* background,
//                          double* value_, int* feature_, int* left_child_, int* right_child_,
//                          double* partition_min_, double* partition_max_, double* result) {
    
//     // Load data instances
//     Matrix<double> X_f = createMatrix<double>(Nx, d, foreground);
//     Matrix<double> X_b = createMatrix<double>(Nz, d, background);

//     // Load tree structure
//     Matrix<double> value = createMatrix<double>(Nt, depth, value_);
//     Matrix<int> feature = createMatrix<int>(Nt, depth, feature_);
//     Matrix<int> left_child  = createMatrix<int>(Nt, depth, left_child_);
//     Matrix<int> right_child = createMatrix<int>(Nt, depth, right_child_);
//     vector<Matrix<double>> partition_min = createTensor<double>(Nt, M, d, partition_min_);
//     vector<Matrix<double>> partition_max = createTensor<double>(Nt, M, d, partition_max_);

//     // Precompute the SHAP weights
//     Matrix<double> W(d, vector<double> (d));
//     compute_W(W);
    
//     Tensor<double> phi = obs_treeSHAP(X_f, X_b, feature, left_child, right_child,
//                                         partition_min, partition_max, value, W, max_var);
//     cout << endl;

//     /// Save the results
//     for (unsigned int i(0); i < phi.size(); i++){
//         for (int j(0); j < Nx; j++){
//             for (int k(0); k < d; k++){
//                 result[i*d*Nx + j*d + k] = phi[i][j][k];
//             }
//         }
//     }
//     return 0;
// }




// ********************************************************************************
//                                   Parameters
// ********************************************************************************
//     int Nx : number of foreground instances
//     int Nt : number of trees
//     int d  : number of input features
//     int depth : max depth of the tree ensemble
//     int M : TODO
//     double* foreground : row major (Nx, d) array of foreground data
//     double* background : row major (Nz, d) array of background data
//     int* I_map : function mapping features to their feature group
//     double* value_: row major (Nt, depth) array of values
//     int* feature : row major (Nt, depth) array of feature index
//     int* left_child : row major (Nt, depth) array of left child index
//     int* right_child : row major (Nt, depth) array of right child index
//     double* partition_min_ : TODO
//     double* partition_max_ : TODO
// ********************************************************************************
extern "C"
int main_add_waterfallshap(int Nx, int Nz, int Nt, int d, int depth, int max_depth,
                         double* foreground, double* background, int* I_map,
                         double* threshold_, double* value_, int* feature_, int* left_child_, int* right_child_,
                         double* n_samples_, double* result) {
    
    // Load data instances
    Matrix<double> X_f = createMatrix<double>(Nx, d, foreground);

    // Load tree structure
    Matrix<double> threshold = createMatrix<double>(Nt, depth, threshold_);
    Matrix<double> value = createMatrix<double>(Nt, depth, value_);
    Matrix<int> feature = createMatrix<int>(Nt, depth, feature_);
    Matrix<int> left_child  = createMatrix<int>(Nt, depth, left_child_);
    Matrix<int> right_child = createMatrix<int>(Nt, depth, right_child_);
    Matrix<double> n_samples = createMatrix<double>(Nt, depth, n_samples_);
    // printMatrix(n_samples);

    waterfall_additive(X_f, Nz, feature, left_child, right_child, value, threshold, n_samples, max_depth, I_map, result);
    cout << endl;

    return 0;
}
