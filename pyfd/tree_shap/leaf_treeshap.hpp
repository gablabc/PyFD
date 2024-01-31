#ifndef __LEAFSHAP
#define __LEAFSHAP

#include <vector>
#include <iostream>
#include <utility>
#include <string>
#include <cstdlib>
#include <iomanip>
#include <stdexcept>
#include <set>
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
                    int* Imap,
                    double* result){
    // Setup
    int d = X_f[0].size();
    int n_trees = feature.size();
    int Nz = X_b.size();
    int Nx = X_f.size();
    int n_features = return_max(Imap, d)+1;

    // Initialize the SHAP values to zero
    progressbar bar(n_trees);

    // Variables for coallitional game
    int inside;
    double N_L;
    vector<int> game_background(n_features, 0);    // #(z in L_{ I^-1({k})bar })
    vector<int> branch(0), branch_I(0);            // vectors of node i and image I(i) of the current branch

    // Variables for the tree traversal
    int leaf_idx, curr_node_index, curr_depth, curr_feature, going_depth_up;
    tuple<int, int, bool> curr_tuple;

    // Iterate over all trees
    for (int t(0); t < n_trees; t++){
        // Iterate over all leafs, and stack tuples
        // tuple<int, bool, int> which represent the node index, its depth, and 
        //                       if its a right child, 
        stack<tuple<int, int, bool>> curr_path;
        FeatureSet S_L(d, Imap);        // Class representation of the set S_L
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
                branch = S_L.get_S_vector();
                branch_I = S_L.get_I_S_vector();

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
                    for (auto & b : branch){
                        if( (X_b[j][b] <= partition_min[t][leaf_idx][b]) || X_b[j][b] > partition_max[t][leaf_idx][b])
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

                // Fill background_game with counts from the background N(z in L_{ I^-1({k})bar })
                for (auto & k : branch_I) {
                    game_background[k] = 0;
                    for (int j(0); j < Nz; j++){
                        inside = 1;
                        // Iterate over all features not in I^-1({k})
                        for (auto & b : branch){
                            if (Imap[b] == k){
                                continue;
                            }
                            if ((X_b[j][b] <= partition_min[t][leaf_idx][b]) || (X_b[j][b] > partition_max[t][leaf_idx][b])){
                                inside = 0;
                                break;
                            }
                        }
                        if (inside){
                            game_background[k] += 1;
                        }
                    }
                }

                // All the points to explain and define the indicator game Ind(x in L_{ I^-1({k}) })
                for (int i(0); i < Nx; i++){
                    for (auto & k : branch_I) {
                        inside = 1;
                        // Iterate over all features in I^-1({k})
                        for (auto & b : branch){
                            if (Imap[b] != k){
                                continue;
                            }
                            if ((X_f[i][b] <= partition_min[t][leaf_idx][b]) || (X_f[i][b] > partition_max[t][leaf_idx][b])){
                                inside = 0;
                                break;
                            }
                        }
                        result[i*n_features + k] += (inside * game_background[k] - N_L) * value[t][curr_node_index] / Nz;
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





// Leaf-based Shapley values
int leaf_treeshap(Matrix<double> &X_f, 
                    Matrix<double> &X_b, 
                    Matrix<int> &feature,
                    Matrix<int> &child_left,
                    Matrix<int> &child_right,
                    Tensor<double> &partition_min,
                    Tensor<double> &partition_max,
                    Matrix<double> &value,
                    int* Imap,
                    int max_var,
                    double* result){
    // Setup
    int d = X_f[0].size();
    int n_trees = feature.size();
    int Nz = X_b.size();
    int Nx = X_f.size();
    int n_features = return_max(Imap, d)+1;
    int max_subsets = integer_exp(2, max_var);

    // Precompute the SHAP weights
    Matrix<double> W(n_features, vector<double> (n_features));
    compute_W(W);
    
    // Initialize the SHAP values to zero
    progressbar bar(n_trees);

    // Variables for coallitional game
    int inside, S_size, powset_size, num_players;
    double nu_S, nu_S_k;                            // nu_I(S) and nu_I(S U {k})
    vector<int> game_background(max_subsets, 0);    // #(z in L_{I^-1(S)bar})
    vector<int> game_foreground(max_subsets, 0);    // Ind(x in L_{I^-1(S)})
    vector<int> branch(0), branch_I(0);             // vectors of node i and image I(i) of the current branch
    
    Matrix<int> in_I_minus_S(max_subsets, vector<int> (d, 0));
    vector<int> S_sizes(max_subsets, 0);

    // Variables for the tree traversal
    int leaf_idx, curr_node_index, curr_depth, curr_feature, going_depth_up;
    tuple<int, int, bool> curr_tuple;

    // Iterate over all trees
    for (int t(0); t < n_trees; t++){
        // Iterate over all leafs, and stack tuples
        // tuple<int, bool, int> which represent the node index, its depth, and 
        //                       if its a right child, 
        stack<tuple<int, int, bool>> curr_path;
        FeatureSet S_L(d, Imap);         // Class representation of the set S_L
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
                branch = S_L.get_S_vector();
                branch_I = S_L.get_I_S_vector();

                // for (int i(0); i < S_L_vector.size(); i++){
                //     cout << S_L_vector[i] << " ";
                // }
                // cout << endl;
                // for (int i(0); i < I_S_L_vector.size(); i++){
                //     cout << I_S_L_vector[i] << " ";
                // }
                // cout << endl;

                // Set the coallitional game
                num_players = branch_I.size();
                powset_size = integer_exp(2, num_players);

                // Iterate over the power set
                // each set S is an integer whose bits are interpreted as the elements inside it
                // e.g. {0, 2, 4} = 00010101
                for (int S(0); S < powset_size; S++){
                    // cout << "S : " << S << endl;
                    S_sizes[S] = 0;
                    game_background[S] = 0;

                    // Count the number of players of each subset e.g. 001101 has three
                    for (int k(0); k < num_players; k++){
                        if ((S & (1 << k)) > 0){
                            S_sizes[S]++;
                        }
                    }
                    // Fill up the S matrix
                    for (auto & b : branch){
                        in_I_minus_S[S][b] = isin_S(Imap[b], S, branch_I);
                    }

                    // Iterate over all background instances to get N(L_{I^{-1}(S)bar})
                    for (int j(0); j < Nz; j++){
                        inside = 1;
                        // Iterate over all features not in I^-1(S)
                        for (auto & b : branch){
                            if (in_I_minus_S[S][b]){
                                continue;
                            }
                            if ((X_b[j][b] <= partition_min[t][leaf_idx][b]) || (X_b[j][b] > partition_max[t][leaf_idx][b])){
                                inside = 0;
                                break;
                            }
                        }
                        if (inside){
                            game_background[S] += 1;
                        }
                    }
                }

                // For all foreground point x define the indicator game Ind(x \in L_{I^{-1}(S)})
                for (int i(0); i < Nx; i++){
                    // Iterate over the power set of S
                    for (int S(0); S < powset_size; S++){
                        inside = 1;
                        // Iterate over all features in I^-1({S})
                        for (auto & b : branch){
                            if (!in_I_minus_S[S][b]){
                                continue;
                            }
                            if ((X_f[i][b] <= partition_min[t][leaf_idx][b]) || (X_f[i][b] > partition_max[t][leaf_idx][b])){
                                inside = 0;
                                break;
                            }
                        }
                        // x lands in the projected leaf
                        game_foreground[S] = inside;
                    }

                    // Compute SHAP value of each feature
                    for (int k(0); k < num_players; k++){
                        // Iterate over the power set of S
                        for (int S(0); S < powset_size; S++){
                            // Only consider coallitions that exclude the player
                            if ((S & (1 << k)) > 0){
                                continue;
                            }
                            S_size = S_sizes[S];
                            nu_S  = game_foreground[S] * game_background[S];
                            nu_S_k  = game_foreground[S ^ (1 << k)] * game_background[S ^ (1 << k)];

                            // Add the marginal contribution
                            result[i*n_features + branch_I[k]] += W[S_size][num_players-1] * 
                                                    (nu_S_k - nu_S) * value[t][curr_node_index] / Nz;
                        }
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

#endif