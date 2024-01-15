#ifndef __STACK
#define __STACK

#include <vector>
#include <iostream>
#include <utility>
#include <string>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <iomanip>
#include <stdexcept>
#include "progressbar.hpp"
#include "utils.hpp"

using namespace std;


// Main function for compute A with a stack
Matrix<double> A_treeSHAP_stack(Matrix<double> &X, 
                                Matrix<int> &feature,
                                Matrix<int> &left_child,
                                Matrix<int> &right_child,
                                Matrix<double> &threshold,
                                Matrix<double> &value)
    {
    // Setup
    int n_features = X[0].size();
    int n_trees = feature.size();
    int N = X.size();

    // Initialize the taylor SHAP values to zero
    Matrix <double> A(N, vector<double> (N, 0));

    progressbar bar(N*(N+1)/2);

    // Init variables for tree-traversal
    int parent_feature, going_depth_up;
    int curr_tag, x_child, z_child;
    tuple<int, int, int, int> curr_tuple;
    FeatureSet Sets(n_features);           // Class representation of the set SX and SZ
    // Traverse the tree via a stack who elements are
    // tuple<int, bool, int> which represent the node index, its depth, parent_feature, and tag
    stack<tuple<int, int, int, int>> candidates;

    // Iterate over all foreground instances
    for (int i(0); i < N; i++){
        // Iterate over all background instances
        for (int j(i); j < N; j++){
            // Iterate over all trees
            for (int t(0); t < n_trees; t++){
                // cout << "|SX| " << Sets.size_SX() << endl;
                // cout << "|SZ| " << Sets.size_SZ() << endl;
                // cout << "Is path reset " << Sets.is_path_empty() << endl;

                // Init Root node
                int n = 0;
                int curr_feature = feature[t][0];
                int curr_depth = 0;
                // cout << "starting" << endl;
                // Explore the whole tree via a stack
                while (true) {

                    // Reached a leaf
                    if (curr_feature < 0){
                        // cout << "leaf" << endl;
                        // cout << "|SX| " << Sets.size_SX() << endl;
                        // cout << "|SZ| " << Sets.size_SZ() << endl;
                        if (Sets.size_SX() > 1 && Sets.size_SZ() > 1){
                            cout << "Error in tree traversal" << endl;
                        }
                        // Diagonal element
                        if (i == j){
                            A[i][i] += value[t][n];
                        }
                        else {
                            // |S_X| = 0 so EACH element of S_Z gets a contribution
                            if (Sets.size_SX()==0){
                                A[i][j] += (1 - Sets.size_SZ()) * value[t][n];
                            }
                            // |S_X| = 1 so the SINGLE element of S_X gets a contribution
                            else if (Sets.size_SX()==1){
                                A[i][j] += value[t][n];
                            }

                            // |S_Z| = 0 so EACH element of S_X gets a contribution
                            if (Sets.size_SZ()==0){
                                A[j][i] += (1 - Sets.size_SX()) * value[t][n];
                            }
                            // |S_Z| = 1 so the SINGLE element of S_Z gets a contribution
                            else if (Sets.size_SZ()==1){
                                A[j][i] += value[t][n];
                            }
                        }
                        // The stack is empty so we are done with traversal
                        if (candidates.empty()){
                            Sets.remove_features(curr_depth);
                            break;
                        }
                        // Otherwise we backtrack
                        going_depth_up = curr_depth - get<1>(candidates.top()) + 1;
                        // cout << "Backtracking " << going_depth_up << " steps" << endl;
                        Sets.remove_features(going_depth_up);
                    }
                    else {
                        // Find children of x and z
                        if (X[i][curr_feature] <= threshold[t][n]){
                            x_child = left_child[t][n];
                        } else {x_child = right_child[t][n];}
                        if (X[j][curr_feature] <= threshold[t][n]){
                            z_child = left_child[t][n];
                        } else {z_child = right_child[t][n];}

                        // Scenario 1 : x and z go the same way so we avoid the type B edge
                        if (x_child == z_child){
                            // cout << "avoid type B" << endl;
                            // Add the feature to the path and keep SX and SZ intact
                            candidates.push(make_tuple(x_child, curr_depth+1, curr_feature, 0));
                        }

                        // Senario 2: x and z go different ways and we have seen this feature i in S_X U S_Z.
                        // Hence we go down the correct edge to ensure that S_X and S_Z are kept disjoint
                        else if (Sets.in_SX(curr_feature) || Sets.in_SZ(curr_feature)){
                            // cout << "Keep SX and SZ disjoint" << endl;
                            // Add the feature to the path and keep SX and SZ intact
                            if (Sets.in_SX(curr_feature)){
                                candidates.push(make_tuple(x_child, curr_depth+1, curr_feature, 0));
                            }
                            else {
                                candidates.push(make_tuple(z_child, curr_depth+1, curr_feature, 0));
                            }
                        }

                        // Scenario 3 : x and z go different ways and we have not yet seen this feature
                        else {
                            // cout << "branching" << endl;
                            // cout << "|SX| " << Sets.size_SX() << endl;
                            // cout << "|SZ| " << Sets.size_SZ() << endl;
                            // Go to z's child if it is allowed and update SZ
                            if (Sets.size_SX() <= 1 || Sets.size_SZ() == 0){
                                // cout << "going down z child" << endl;
                                candidates.push(make_tuple(z_child, curr_depth+1, curr_feature, 2));
                            }

                            // Go to x's child if it is allowed and update SX
                            if (Sets.size_SX() == 0 || Sets.size_SZ() <= 1){
                                // cout << "going down x child" << endl;
                                candidates.push(make_tuple(x_child, curr_depth+1, curr_feature, 1));
                            }
                        }
                    }

                    // Pop the triplet on top of the stack
                    curr_tuple = candidates.top();
                    candidates.pop();
                    n = get<0>(curr_tuple);
                    curr_depth = get<1>(curr_tuple);
                    parent_feature = get<2>(curr_tuple);
                    curr_tag = get<3>(curr_tuple);
                    // cout << curr_depth << endl;
                    // cout << parent_feature << endl;
                    // cout << curr_tag << endl;

                    // Update the feature path
                    Sets.add_feature(parent_feature, curr_tag);

                    // Set the feature of the current node
                    curr_feature = feature[t][n];
                }
            }
        bar.update();
        }
    }
    return A;
}

# endif