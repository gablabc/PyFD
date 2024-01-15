#ifndef __WATERFALLSHAP
#define __WATERFALLSHAP

#include <vector>
#include <iostream>
#include <utility>
#include <string>
#include <cstdlib>
#include <iomanip>
#include <stdexcept>
#include "progressbar.hpp"
#include "utils.hpp"


// Waterfall-based additive decomposition
int waterfall_additive(Matrix<double> &X_f,
                    int Nz,
                    Matrix<int> &feature,
                    Matrix<int> &child_left,
                    Matrix<int> &child_right,
                    Matrix<double> &value,
                    Matrix<double> &threshold,
                    Matrix<double> &n_samples,
                    int max_depth,
                    int* I_map,
                    double* result){
    // Setup
    int d = X_f[0].size();
    int n_trees = feature.size();
    int Nx = X_f.size();
    int n_features = return_max(I_map, d)+1;

    // Initialize the SHAP values to zero
    progressbar bar(Nx);

    // Waterfall weights
    double weights[max_depth+1][n_features];
    // Matrix<double> weights (max_depth+1, vector<double> (n_features));
    for (int j(0); j < n_features; j++){
        weights[0][j] = (double) Nz;
    }

    // Variables for the tree traversal
    double ratio;
    int leaf_idx, curr_node, curr_depth, curr_parent, curr_parent_feature, is_right_child;
    tuple<int, int, int, bool> curr_tuple;

    // Iterate over foreground
    for (int i(0); i < Nx; i++){
        // cout << "Explaining instance : " << i << endl;
        // for (int k(0); k < n_features; k++){
        //     cout << X_f[i][k] << " ";
        // }
        // cout << endl;

        // Iterate over all trees
        for (int t(0); t < n_trees; t++){
            // Iterate over all leafs, and stack tuples
            // tuple<int, int, int, bool> which represent the node index, its depth,
            //                       its parent index, and if its a right child, 
            stack<tuple<int, int, int, bool>> curr_path;
            // Push both root's children to the stack
            curr_path.push(make_tuple(child_right[t][0], 1, 0, true));
            curr_path.push(make_tuple(child_left[t][ 0], 1, 0, false));

            // Explore the whole tree via a stack
            while ( !curr_path.empty() ){
                // Pop the tuple on top of the stack
                curr_tuple = curr_path.top();
                curr_path.pop();
                curr_node= get<0>(curr_tuple);
                curr_depth = get<1>(curr_tuple);
                curr_parent = get<2>(curr_tuple);
                is_right_child = get<3>(curr_tuple);
                curr_parent_feature = feature[t][curr_parent];
                // cout << "Visiting node " << curr_node << endl;
                // cout << "Depth " << curr_depth << endl;
                // cout << "Parent " << curr_parent << endl;
                // cout << "Parent Feature " << curr_parent_feature << endl;
                // cout << "Right child? " << is_right_child << endl;

                // Update the weights at that current depth
                for (int k(0); k < n_features; k++){
                    // cout << "Split along " << k << " : ";
                    // Multiply previous weight by 0/1
                    if (I_map[curr_parent_feature] == k){
                        if (is_right_child){
                            // cout << "Flow right ";
                            ratio = (double)(X_f[i][curr_parent_feature] > threshold[t][curr_parent]);
                        }
                        else{
                            // cout << "Flow left ";
                            ratio = (double)(X_f[i][curr_parent_feature] <= threshold[t][curr_parent]);
                        }
                    }
                    // Multiply previous weights by ratio going left and right
                    else {
                        // cout << n_samples[t][curr_node] << " / " << n_samples[t][curr_parent] << " = ";
                        ratio = n_samples[t][curr_node] / n_samples[t][curr_parent];
                    }
                    // cout << ratio << endl;
                    weights[curr_depth][k] = weights[curr_depth-1][k] * ratio;
                }
                // printMatrix(weights);

                // Not at a leaf yet
                if (feature[t][curr_node] >= 0){
                    // Push both children to the stack
                    curr_path.push(make_tuple(child_right[t][curr_node], curr_depth+1, curr_node, true));
                    curr_path.push(make_tuple(child_left[t][ curr_node], curr_depth+1, curr_node, false));
                }
                // At a leaf
                else {
                    // cout << "Reached a leaf" << endl;
                    for (int k(0); k < n_features; k++) {
                        result[i*n_features + k] += (weights[curr_depth][k] - n_samples[t][curr_node]) * value[t][curr_node] / Nz;
                    }
                }
            }
        }
        bar.update();
    }
    return 0;
}

#endif