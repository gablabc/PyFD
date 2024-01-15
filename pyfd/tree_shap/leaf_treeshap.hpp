#ifndef __LEAFSHAP
#define __LEAFSHAP

#include <vector>
#include <iostream>
#include <utility>
#include <string>
#include <cstdlib>
#include <iomanip>
#include <stdexcept>
#include "progressbar.hpp"
#include "utils.hpp"


// Leaf-based additive decomposition
int leaf_additive(Matrix<double> &X_f, 
                    Matrix<double> &X_b, 
                    Matrix<int> &feature,
                    Matrix<int> &child_left,
                    Matrix<int> &child_right,
                    Tensor<double> &partition_min,
                    Tensor<double> &partition_max,
                    Matrix<double> &value,
                    int* I_map,
                    double* result){
    // Setup
    int d = X_f[0].size();
    int n_trees = feature.size();
    int Nz = X_b.size();
    int Nx = X_f.size();
    int n_features = return_max(I_map, d)+1;

    // Initialize the SHAP values to zero
    progressbar bar(n_trees);

    // Variables for coallitional game
    int inside;
    double N_L;
    vector<int> N_L_game(n_features, 0);   // N(L_{I^-1{i}bar})
    vector<int> S_L_vector(0), I_S_L_vector(0);

    // Variables for the tree traversal
    int leaf_idx, curr_node_index, curr_depth, curr_feature, going_depth_up;
    tuple<int, int, bool> curr_tuple;

    // Iterate over all trees
    for (int t(0); t < n_trees; t++){
        // Iterate over all leafs, and stack tuples
        // tuple<int, bool, int> which represent the node index, its depth, and 
        //                       if its a right child, 
        stack<tuple<int, int, bool>> curr_path;
        FeatureSet S_L(d, I_map);        // Class representation of the set S_L
        leaf_idx = 0;                    // Each time we visit a leaf we increment by one
        curr_path.push(make_tuple(0, 0, false));

        // Explore the whole tree via a stack
        while ( !curr_path.empty() ){
            // Pop the triplet on top of the stack
            curr_tuple = curr_path.top();
            curr_path.pop();
            curr_node_index = get<0>(curr_tuple);
            curr_depth = get<1>(curr_tuple);
            curr_feature = feature[t][curr_node_index];

            // Not at a leaf yet
            if (curr_feature >= 0){
                // Add the feature to the set S_L
                S_L.add_feature(curr_feature);
                curr_path.push(make_tuple(child_right[t][curr_node_index], curr_depth+1, true));
                curr_path.push(make_tuple(child_left[t][ curr_node_index], curr_depth+1, false));
            }
            // At a leaf
            else {
                // cout << "Reached leaf " << leaf_idx << endl;

                // Vectors of features involved in the game
                S_L_vector = S_L.get_S_vector();
                I_S_L_vector = S_L.get_I_S_vector();

                // for (int i(0); i < S_L_vector.size(); i++){
                //     cout << S_L_vector[i] << " ";
                // }
                // cout << endl;
                // for (int i(0); i < I_S_L_vector.size(); i++){
                //     cout << I_S_L_vector[i] << " ";
                // }
                // cout << endl;

                N_L = 0;
                // Count over the background data to compute N_L
                for (int j(0); j < Nz; j++){
                    inside = 1;
                    // Across all dimensions, a point must be in the interval
                    for (auto & s : S_L_vector){
                        if( (X_b[j][s] <= partition_min[t][leaf_idx][s]) || X_b[j][s] > partition_max[t][leaf_idx][s])
                        {
                            inside = 0;
                            break;
                        }
                    }
                    if (inside){
                        N_L = N_L + 1;
                    }
                }
                // cout << "Contains " << N_L << " datapoints\n";

                // Fill-out the vector N_L_game with counts from the background
                for (auto & I_k : I_S_L_vector) {
                    N_L_game[I_k] = 0;
                    for (int j(0); j < Nz; j++){
                        inside = 1;
                        // Iterate over all features not in I^-1({k})
                        for (auto & s : S_L_vector){
                            if (I_map[s] == I_k){
                                continue;
                            }
                            if ((X_b[j][s] <= partition_min[t][leaf_idx][s]) || (X_b[j][s] > partition_max[t][leaf_idx][s])){
                                inside = 0;
                                break;
                            }
                        }
                        if (inside){
                            N_L_game[I_k] += 1;
                        }
                    }
                }

                // All the points to explain and define the indicator game
                for (int i(0); i < Nx; i++){
                    for (auto & I_k : I_S_L_vector) {
                        inside = 1;
                        // Iterate over all features in I^-1({k})
                        for (auto & s : S_L_vector){
                            if (I_map[s] != I_k){
                                continue;
                            }
                            if ((X_f[i][s] <= partition_min[t][leaf_idx][s]) || (X_f[i][s] > partition_max[t][leaf_idx][s])){
                                inside = 0;
                                break;
                            }
                        }
                        result[i*n_features + I_k] += (inside * N_L_game[I_k] - N_L) * value[t][curr_node_index] / Nz;
                    }           
                }
                // If we are a right child, then we must remove features from S_L since we
                // are backtraking through the tree
                if ( get<2>(curr_tuple) && !curr_path.empty()){
                    going_depth_up = get<1>(curr_tuple) - get<1>(curr_path.top());
                    S_L.remove_features(going_depth_up);
                }
                leaf_idx += 1;
            }
        }
        bar.update();
    }
    return 0;
}


// // Main function for TreeSHAP
// Tensor<double> obs_treeSHAP(Matrix<double> &X_f, 
//                             Matrix<double> &X_b, 
//                             Matrix<int> &feature,
//                             Matrix<int> &child_left,
//                             Matrix<int> &child_right,
//                             Tensor<double> &partition_min,
//                             Tensor<double> &partition_max,
//                             Matrix<double> &value,
//                             Matrix<double> &W,
//                             int max_var) {
//     // Setup
//     int d = X_f[0].size();
//     int n_trees = feature.size();
//     int Nz = X_b.size();
//     int Nx = X_f.size();

//     // Initialize the SHAP values to zero
//     Tensor<double> phi_f_b(n_trees, Matrix<double>(Nx, vector<double> (d, 0)));
//     progressbar bar(n_trees);

//     int count, pow_set_size, S_size;
//     double lm, phi, p_s, p_si;

//     int max_players = integer_exp(2, max_var);
//     Matrix<int> S(d, vector<int>(max_players, 0));
//     vector<int> lm_s(max_players, 0);   // Leaf game
//     vector<int> Ix_s(max_players, 0);   // Indicator game
//     vector<int> S_sizes(max_players, 0);
//     vector<int> feature_in_game (0);

//     // Variable for the tree traversal
//     int curr_node_index, curr_depth, curr_feature, going_depth_up;
//     tuple<int, int, bool> curr_tuple;

//     // Iterate over all trees
//     for (int t(0); t < n_trees; t++){
//         // Iterate over all leafs, and stack tuples
//         // tuple<int, bool, int> which represent the node index, its depth, and 
//         //                       if its a right child, 
//         stack<tuple<int, int, bool>> curr_path;
//         FeatureSet S_L(d);        // Class representation of the set S_L
//         int leaf_idx(0);          // Each time we visit a leaf we increment by one
//         curr_path.push(make_tuple(0, 0, false));

//         // Explore the whole tree via a stack
//         while ( !curr_path.empty() ){
//             // Pop the triplet on top of the stack
//             curr_tuple = curr_path.top();
//             curr_path.pop();
//             curr_node_index = get<0>(curr_tuple);
//             curr_depth = get<1>(curr_tuple);
//             curr_feature = feature[t][curr_node_index];

//             // Not at a leaf yet
//             if (curr_feature >= 0){
//                 // Add the feature to the set S_L
//                 S_L.add_feature(curr_feature);
//                 curr_path.push(make_tuple(child_right[t][curr_node_index], curr_depth+1, true));
//                 curr_path.push(make_tuple(child_left[t][ curr_node_index], curr_depth+1, false));
//             }
//             // At a leaf
//             else {
//                 lm = 0;
//                 // Count over the background data to compute N_L
//                 for (int i(0); i < Nz; i++){
//                     count = 0;
//                     // Across all dimensions, a point must be in the interval
//                     for (int j(0); j < d; j++){
//                         if( (X_b[i][j] >  partition_min[t][leaf_idx][j]) && 
//                              X_b[i][j] <= partition_max[t][leaf_idx][j])
//                         {
//                             count += 1;
//                         }
//                     }
//                     if (count == d){
//                         lm = lm + 1;
//                     }
//                 }

//                 // The list of feature involved in the game
//                 feature_in_game = S_L.get_feature_vector();
//                 // The number of subsets of players we must consider
//                 pow_set_size = integer_exp(2, S_L.size());

//                 // Iterate over the power set of S to define the leaf game
//                 for (int counter(0); counter < pow_set_size; counter++){
//                     S_size = 0;
//                     lm_s[counter] = 0;

//                     // Create a mapping S between the subset of active players 
//                     // defined as an integer 001101 and the actual feature
//                     for (unsigned int f(0); f < feature_in_game.size(); f++){
//                         // f is also present in the subset S
//                         if ((counter & (1 << f)) > 0){
//                             S[S_size][counter] = feature_in_game[f];
//                             S_size += 1;
//                         }
//                     }
//                     S_sizes[counter] = S_size;

//                     // Iterate over all background instances to count N_L_S
//                     for (int k(0); k < Nz; k++){
//                         count = 0;
//                         for (int s(0); s < S_size; s++){
//                             // Is x_S in L_S??
//                             if ((X_b[k][S[s][counter]] >
//                                 partition_min[t][leaf_idx][S[s][counter]]) && 
//                                 (X_b[k][S[s][counter]] <=
//                                 partition_max[t][leaf_idx][S[s][counter]])){
//                                 count += 1;
//                             }
//                         }
//                         // x_S is part of L_S
//                         if (count == S_size){
//                             lm_s[counter] += 1;
//                         }
//                     }
//                 }

//                 // All the points to explain and define the indicator game I(x_s \in L_S)
//                 for (int k(0); k < Nx; k++){
//                     // Iterate over the power set of S
//                     for (int counter(0); counter < pow_set_size; counter++){
//                         S_size = S_sizes[counter];
//                         count = 0;

//                         // Check if x_S lands in L_S
//                         for (int s(0); s < S_size; s++){
//                             // If x_S in L_S??
//                             if ((X_f[k][S[s][counter]] >
//                             partition_min[t][leaf_idx][S[s][counter]]) && 
//                             (X_f[k][S[s][counter]] <=
//                             partition_max[t][leaf_idx][S[s][counter]])){
//                                 count += 1;
//                             }
//                         }

//                         // x_S lands in the leaf
//                         if (count == S_size){
//                             Ix_s[counter] = 1;
//                         }
//                         else { Ix_s[counter] = 0;}
//                     }

//                     // Compute SHAP value of each in-game feature individually
//                     for (int i(0); i < S_L.size(); i++){

//                         // Iterate over the power set of S
//                         for (int counter(0); counter < pow_set_size; counter++){
//                             // Only consider coallitions that exclude {i}
//                             if ((counter & (1 << i)) > 0){
//                                 continue;
//                             }
//                             S_size = S_sizes[counter];

//                             // nu(S)
//                             if ( lm_s[counter] !=0 ){
//                                 p_s = (Ix_s[counter] * lm) / lm_s[counter];
//                             }
//                             else { p_s = 0; }
//                             // nu(S U {i})
//                             if ( lm_s[counter ^ (1 << i)] !=0 ){
//                                 p_si = (Ix_s[counter ^ (1 << i)] * lm) / lm_s[counter ^ (1 << i)];
//                             }
//                             // WHY WOULD LM_S be zero? Since each leaf must have data inside it??? 
//                             else { p_si = 0; };
//                             // Add the marginal contribution
//                             phi = W[S_size][S_L.size()-1] * (p_si - p_s) * value[t][curr_node_index];
//                             phi_f_b[t][k][feature_in_game[i]] += phi;
//                         }
//                     }           
//                 }
//                 // If we are a right child, then we must remove features from S_L since we
//                 // are backtraking through the tree
//                 if ( get<2>(curr_tuple) && !curr_path.empty()){
//                     going_depth_up = get<1>(curr_tuple) - get<1>(curr_path.top());
//                     S_L.remove_features(going_depth_up);
//                 }
//                 leaf_idx += 1;
//             }
//         }
//         bar.update();
//     }
//     return phi_f_b;
// }

#endif