#ifndef __RECURS
#define __RECURS

#include <vector>
#include <iostream>
#include <utility>
#include <string>
#include <cstdlib>
#include <iomanip>
#include <stdexcept>
#include "progressbar.hpp"
#include "utils.hpp"

// void recurse_treeshap(int n, double* x, double* z, 
//                     struct TreeEnsemble* tree,
//                     int* I_map,
//                     int& n_features,
//                     vector<int> &in_ISX,
//                     vector<int> &in_ISZ,
//                     Matrix<double> &W,
//                     int* current_depth,
//                     struct UpwardFlow* flow,
//                     vector<double> &phi)
// {
//     // cout << "Node " << n << " at depth : " << *current_depth << endl;
//     int current_feature = tree->feature[n];
//     int x_child(0), z_child(0);
//     // num_players := |I(S_{XZ})|
//     int num_players = 0;

//     // Arriving at a Leaf
//     if (tree->right_child[n] < 0)
//     {
//         double pos(0.0), neg(0.0);
//         num_players = in_ISX[n_features] + in_ISZ[n_features];
//         if (in_ISX[n_features] > 0)
//         {
//             pos = W[in_ISX[n_features]-1][num_players-1] * tree->value[n];
//         }
//         if (in_ISZ[n_features] > 0)
//         {
//             neg = W[in_ISX[n_features]][num_players-1] * tree->value[n];
//         }
//         // cout << " Computing outflow at depth " << *current_depth << endl;
//         flow->pos[*current_depth] = pos;
//         flow->neg[*current_depth] = neg;
//     }
//     else {
//         // Find children of x and z
//         x_child = x[current_feature] <= tree->threshold[n] ? tree->left_child[n] : tree->right_child[n];
//         z_child = z[current_feature] <= tree->threshold[n] ? tree->left_child[n] : tree->right_child[n];

//         // Scenario 1 : x and z go the same way so we avoid the type B edge
//         if (x_child == z_child){
//             // cout << "Scenario 1" << endl;
//             *current_depth += 1;
//             recurse_treeshap(x_child, x, z, tree, I_map, n_features, in_ISX, in_ISZ, W, current_depth, flow, phi);
//             *current_depth -= 1;
//             // cout << "flow going up from " << *current_depth+1 << " to " << *current_depth << endl;
//             flow->pos[*current_depth] = flow->pos[*current_depth+1];
//             flow->neg[*current_depth] = flow->neg[*current_depth+1];
//             return;
//         }

//         // Scenario 2: x and z go different ways and we have seen this feature I(i) in I(S_X) U I(S_Z).
//         // Hence, we go down the correct edge to ensure that I(S_X) and I(S_Z) are kept disjoint
//         if (in_ISX[ I_map[current_feature] ] || in_ISZ[ I_map[current_feature] ]){
//             // cout << "Scenario 2" << endl;
//             if (in_ISX[ I_map[current_feature] ]){
//                 *current_depth += 1;
//                 recurse_treeshap(x_child, x, z, tree, I_map, n_features, in_ISX, in_ISZ, W, current_depth, flow, phi);
//                 *current_depth -= 1;
//                 // cout << "flow going up from " << *current_depth + 1 << " to " << *current_depth << endl;
//                 flow->pos[*current_depth] = flow->pos[*current_depth+1];
//                 flow->neg[*current_depth] = flow->neg[*current_depth+1];
//                 return;
//             }
//             else{
//                 *current_depth += 1;
//                 recurse_treeshap(z_child, x, z, tree, I_map, n_features, in_ISX, in_ISZ, W, current_depth, flow, phi);
//                 *current_depth -= 1;
//                 // cout << "flow going up from " << *current_depth + 1 << " to " << *current_depth << endl;
//                 flow->pos[*current_depth] = flow->pos[*current_depth+1];
//                 flow->neg[*current_depth] = flow->neg[*current_depth+1];
//                 return;
//             }
//         }

//         // Scenario 3 : x and z go different ways and we have not yet seen this feature
//         else {
//             // cout << "Scenario 3" << endl;
//             // Initialize the upward flow from this node
//             flow->pos[*current_depth] = 0.0;
//             flow->neg[*current_depth] = 0.0;

//             // Go to x's child
//             in_ISX[ I_map[current_feature] ]++;
//             in_ISX[n_features]++;
//             *current_depth += 1;
//             recurse_treeshap(x_child, x, z, tree, I_map, n_features, in_ISX, in_ISZ, W, current_depth, flow, phi);
//             *current_depth -= 1;
//             in_ISX[ I_map[current_feature] ]--;
//             in_ISX[n_features]--;
//             // Update the flow
//             // cout << "flow going up from " << *current_depth+1 << " to " << *current_depth << endl;
//             flow->pos[*current_depth] += flow->pos[*current_depth + 1];
//             flow->neg[*current_depth] += flow->neg[*current_depth + 1];
//             // Add incoming positive flow to the current feature
//             phi[ I_map[current_feature] ] += flow->pos[*current_depth + 1];
            
//             // Go to z's child
//             in_ISZ[ I_map[current_feature] ]++;
//             in_ISZ[n_features]++;
//             *current_depth += 1;
//             recurse_treeshap(z_child, x, z, tree, I_map, n_features, in_ISX, in_ISZ, W, current_depth, flow, phi);
//             *current_depth -= 1;
//             in_ISZ[ I_map[current_feature] ]--;
//             in_ISZ[n_features]--;
//             // Update the flow
//             // cout << "flow going up from " << *current_depth+1 << " to " << *current_depth << endl;
//             flow->pos[*current_depth] += flow->pos[*current_depth + 1];
//             flow->neg[*current_depth] += flow->neg[*current_depth + 1];

//             // Add incoming negative flow to the current feature
//             phi[ I_map[current_feature] ] -= flow->neg[*current_depth + 1];
//             return;
//         }
//     }
// }

// Recursion function for treeSHAP
// The output pair is interpreted as follows
// pair.first -> positive contribution going to I(S_X)
// pair.second -> negative contribution going to I(S_Z)
pair<double, double> recurse_treeshap(int n, double* x, double* z, 
                                struct TreeEnsemble* tree,
                                struct Image* image,
                                Matrix<double> &W,
                                double* phi)
{
    
    int current_feature = tree->feature[n];
    int x_child(0), z_child(0);
    // num_players := |I(S_{XZ})|
    int num_players = 0;

    // Arriving at a Leaf
    if (tree->right_child[n] < 0)
    {
        double pos(0.0), neg(0.0);
        num_players = image->card_ISX + image->card_ISZ;
        if (image->card_ISX > 0)
        {
            pos = W[image->card_ISX-1][num_players-1] * tree->value[n];
        }
        if (image->card_ISZ > 0)
        {
            neg = W[image->card_ISX][num_players-1] * tree->value[n];
        }
        return make_pair(pos, neg);
    }
    
    // Find children of x and z
    x_child = x[current_feature] <= tree->threshold[n] ? tree->left_child[n] : tree->right_child[n];
    z_child = z[current_feature] <= tree->threshold[n] ? tree->left_child[n] : tree->right_child[n];

    // Scenario 1 : x and z go the same way so we avoid the type B edge
    if (x_child == z_child){
        return recurse_treeshap(x_child, x, z, tree, image, W, phi);
    }

    // Scenario 2: x and z go different ways and we have seen this feature I(i) in I(S_X) U I(S_Z).
    // Hence, we go down the correct edge to ensure that I(S_X) and I(S_Z) are kept disjoint
    if (image->in_ISX[ image->Imap[current_feature] ] || image->in_ISZ[ image->Imap[current_feature] ]){
        if (image->in_ISX[ image->Imap[current_feature] ]){
            return recurse_treeshap(x_child, x, z, tree, image, W, phi);
        }
        else{
            return recurse_treeshap(z_child, x, z, tree, image, W, phi);
        }
    }

    // Scenario 3 : x and z go different ways and we have not yet seen this feature
    else {
        // Go to x's child
        image->in_ISX[ image->Imap[current_feature] ]++;
        image->card_ISX++;
        pair<double, double> pairf = recurse_treeshap(x_child, x, z, tree, image, W, phi);
        image->in_ISX[ image->Imap[current_feature] ]--;
        image->card_ISX--;

        // Go to z's child
        image->in_ISZ[ image->Imap[current_feature] ]++;
        image->card_ISZ++;
        pair<double, double> pairb = recurse_treeshap(z_child, x, z, tree, image, W, phi);
        image->in_ISZ[ image->Imap[current_feature] ]--;
        image->card_ISZ--;

        // Add contribution to the feature
        phi[ image->Imap[current_feature] ] += pairf.first - pairb.second;

        return make_pair(pairf.first + pairb.first, pairf.second + pairb.second);
    }
}



// Recursion function additive components in general case
// The output pair is interpreted as follows
// pair.first -> positive contribution going to I(S_X) when |I(S_X)|=1
// pair.second -> negative contribution going to I(S_Z) when |I(S_X)|=0
pair<double, double> recurse_additive(int n, double* x, double* z, 
                                    struct TreeEnsemble* tree,
                                    struct Image* image,
                                    double* result){
    
    int current_feature = tree->feature[n];
    int x_child(0), z_child(0);

    // Arriving at a Leaf
    if (tree->left_child[n] < 0)
    {   
        double pos(0.0), neg(0.0);
        // |I(S_X)| = 0 so EACH element of I(S_Z) gets a contribution
        if (image->card_ISX==0){
            neg = tree->value[n];
        }
        // |I(S_X)| = 1 so the SINGLE element of I(S_X) gets a contribution
        else if (image->card_ISX==1){
            pos = tree->value[n];
        }

        return make_pair(pos, neg);
    }

    // Find children of x and z
    x_child = x[current_feature] <= tree->threshold[n] ? tree->left_child[n] : tree->right_child[n];
    z_child = z[current_feature] <= tree->threshold[n] ? tree->left_child[n] : tree->right_child[n];

    // Scenario 1 : x and z go the same way so we avoid the type B edge
    if (x_child == z_child){
        return recurse_additive(x_child, x, z, tree, image, result);
    }

    // Scenario 2: x and z go different ways and we have seen this feature I(i) in I(S_X) U I(S_Z).
    // Hence, we go down the correct edge to ensure that I(S_X) and I(S_Z) are kept disjoint
    if (image->in_ISX[ image->Imap[current_feature] ] || image->in_ISZ[ image->Imap[current_feature] ]){
        if (image->in_ISX[ image->Imap[current_feature] ]){
            return recurse_additive(x_child, x, z, tree, image, result);
        }
        else{
            return recurse_additive(z_child, x, z, tree, image, result);
        }
    }

    // Scenario 3 : x and z go different ways and we have not yet seen this feature
    else {
        pair<double, double> pairf, pairb;
        // Go to x's child if it ensures that |I(S_X)|<=1
        if (image->card_ISX == 0){
            image->in_ISX[ image->Imap[current_feature] ]++;
            image->card_ISX++;
            pairf = recurse_additive(x_child, x, z, tree, image, result);
            image->in_ISX[ image->Imap[current_feature] ]--; 
            image->card_ISX--;
        }
        else {
            pairf = make_pair(0.0, 0.0);
        }

        // Go to z's child
        image->in_ISZ[ image->Imap[current_feature] ]++;
        image->card_ISZ++;
        pairb = recurse_additive(z_child, x, z, tree, image, result);
        image->in_ISZ[ image->Imap[current_feature] ]--; 
        image->card_ISZ--;
        

        // Add contribution to the feature
        result[ image->Imap[current_feature] ] += pairf.first - pairb.second;

        return make_pair(pairf.first + pairb.first, pairf.second + pairb.second);
    }
}




// Recursion function additive components in symmetric case
int recurse_additive_sym(int n, double* x, double* z, 
                        struct TreeEnsemble* tree,
                        struct Image* image,
                        vector<int> &ISX,
                        vector<int> &ISZ,
                        double* result_1,
                        double* result_2
                        )
{
    int current_feature = tree->feature[n];
    int x_child(0), z_child(0);

    // Arriving at a Leaf
    if (tree->left_child[n] < 0)
    {   

        // |I(S_X)| = 0 so EACH element of I(S_Z) gets a contribution
        if (image->card_ISX==0){
            for (auto & i : ISZ){
                result_1[i] -= tree->value[n];
            }
        }
        // |S_X| = 1 so the SINGLE element of S_X gets a contribution
        else if (image->card_ISX==1){
            result_1[ISX[0]] += tree->value[n];
        }

        // |S_Z| = 0 so EACH element of S_X gets a contribution
        if (image->card_ISZ==0){
            for (auto & i : ISX){
                result_2[i] -= tree->value[n];
            }
        }
        // |S_Z| = 1 so the SINGLE element of S_Z gets a contribution
        else if (image->card_ISZ==1){
            result_2[ISZ[0]] += tree->value[n];
        }
        return 0;
    }

    // Find children of x and z
    x_child = x[current_feature] <= tree->threshold[n] ? tree->left_child[n] : tree->right_child[n];
    z_child = z[current_feature] <= tree->threshold[n] ? tree->left_child[n] : tree->right_child[n];

    // Scenario 1 : x and z go the same way so we avoid the type B edge
    if (x_child == z_child){
        return recurse_additive_sym(x_child, x, z, tree, image, ISX, ISZ, result_1, result_2);
    }

    // Scenario 2: x and z go different ways and we have seen this feature I(i) in I(S_X) U I(S_Z).
    // Hence, we go down the correct edge to ensure that I(S_X) and I(S_Z) are kept disjoint
    if (image->in_ISX[ image->Imap[current_feature] ] || image->in_ISZ[ image->Imap[current_feature] ]){
        if (image->in_ISX[ image->Imap[current_feature] ]){
            return recurse_additive_sym(x_child, x, z, tree, image, ISX, ISZ, result_1, result_2);
        }
        else{
            return recurse_additive_sym(z_child, x, z, tree, image, ISX, ISZ, result_1, result_2);
        }
    }

    // Scenario 3 : x and z go different ways and we have not yet seen this feature
    else {

        // Go to x's child if it ensures that |I(S_X)|<=1 or |I(S_Z)|<=1
        if (image->card_ISX == 0 || image->card_ISZ <= 1){
            image->in_ISX[ image->Imap[current_feature] ]++; 
            image->card_ISX++;
            ISX.push_back(image->Imap[current_feature]);
            recurse_additive_sym(x_child, x, z, tree, image, ISX, ISZ, result_1, result_2);
            image->in_ISX[ image->Imap[current_feature] ]--;
            image->card_ISX--;
            ISX.pop_back();
        }

        // Go to z's child if it ensures that |I(S_X)|<=1 or |I(S_Z)|<=1
        if (image->card_ISZ == 0 || image->card_ISX <= 1){
            image->in_ISZ[ image->Imap[current_feature] ]++;
            image->card_ISZ++;
            ISZ.push_back(image->Imap[current_feature]);
            recurse_additive_sym(z_child, x, z, tree, image, ISX, ISZ, result_1, result_2);
            image->in_ISZ[ image->Imap[current_feature] ]--;
            image->card_ISZ--;
            ISZ.pop_back();
        }
        return 0;
    }
}





// // Recursion function additive components in symmetric case
// int recurse_additive_sym(int n,
//                         double* x, double* z, 
//                         int* I_map,
//                         int* feature,
//                         int* child_left,
//                         int* child_right,
//                         double* threshold,
//                         double* value,
//                         int n_features,
//                         vector<int> &in_ISX,
//                         vector<int> &in_ISZ,
//                         double* result_1,
//                         double* result_2
//                         )
// {
//     int current_feature = feature[n];
//     int x_child(0), z_child(0);

//     // Arriving at a Leaf
//     if (child_left[n] < 0)
//     {   

//         // |S_X| = 0 so EACH element of S_Z gets a contribution
//         if (in_ISX[n_features]==0){
//             int k(0), counter(0);
//             while (counter < in_ISZ[n_features]){
//                 if (in_ISZ[k]){
//                     result_1[k] -= value[n];
//                     counter++;
//                 }
//                 k++;
//             }
//         }
//         // |S_X| = 1 so the SINGLE element of S_X gets a contribution
//         else if (in_ISX[n_features]==1){
//             int k(0);
//             while (in_ISX[k] == 0){
//                 k++;
//             }
//             result_1[k] += value[n];
//         }

//         // |S_Z| = 0 so EACH element of S_X gets a contribution
//         if (in_ISZ[n_features]==0){
//             int k(0), counter(0);
//             while (counter < in_ISX[n_features]){
//                 if (in_ISX[k]){
//                     result_2[k] -= value[n];
//                     counter++;
//                 }
//                 k++;
//             }
//         }
//         // |S_Z| = 1 so the SINGLE element of S_Z gets a contribution
//         else if (in_ISZ[n_features]==1){
//             int k(0);
//             while (in_ISZ[k] == 0){
//                 k++;
//             }
//             result_2[k] += value[n];
//         }
//         return 0;
//     }

//     // Find children of x and z
//     x_child = x[current_feature] <= threshold[n] ? child_left[n] : child_right[n];
//     z_child = z[current_feature] <= threshold[n] ? child_left[n] : child_right[n];

//     // Scenario 1 : x and z go the same way so we avoid the type B edge
//     if (x_child == z_child){
//         return recurse_additive_sym(x_child, x, z, I_map, feature, child_left, child_right,
//                             threshold, value, n_features, in_ISX, in_ISZ, result_1, result_2);
//     }

    // Scenario 2: x and z go different ways and we have seen this feature I(i) in I(S_X) U I(S_Z).
    // Hence, we go down the correct edge to ensure that I(S_X) and I(S_Z) are kept disjoint
//     if (in_ISX[ I_map[current_feature] ] || in_ISZ[ I_map[current_feature] ]){
//         if (in_ISX[ I_map[current_feature] ]){
//             return recurse_additive_sym(x_child, x, z, I_map, feature, child_left, child_right,
//                             threshold, value, n_features, in_ISX, in_ISZ, result_1, result_2);
//         }
//         else{
//             return recurse_additive_sym(z_child,x, z, I_map, feature, child_left, child_right,
//                             threshold, value, n_features, in_ISX, in_ISZ, result_1, result_2);
//         }
//     }

//     // Scenario 3 : x and z go different ways and we have not yet seen this feature
//     else {

//         // Go to x's child if it ensures that |I(S_X)|<=1 or |I(S_Z)|<=1
//         if (in_ISX[n_features] == 0 || in_ISZ[n_features] <= 1){
//             in_ISX[ I_map[current_feature] ]++; in_ISX[n_features]++;
//             recurse_additive_sym(x_child, x, z, I_map, 
//                                                 feature, child_left, child_right,
//                                                 threshold, value, n_features, in_ISX, in_ISZ, 
//                                                 result_1, result_2);
//             in_ISX[ I_map[current_feature] ]--; in_ISX[n_features]--;
//         }

//         // Go to z's child if it ensures that |I(S_X)|<=1 or |I(S_Z)|<=1
//         if (in_ISZ[n_features] == 0 || in_ISX[n_features] <= 1){
//             in_ISZ[ I_map[current_feature] ]++; in_ISZ[n_features]++;
//             recurse_additive_sym(z_child,x, z, I_map, 
//                                                 feature, child_left, child_right,
//                                                 threshold, value, n_features, in_ISX, in_ISZ,
//                                                 result_1, result_2);
//             in_ISZ[ I_map[current_feature] ]--; in_ISZ[n_features]--;
//         }
//         return 0;
//     }
// }



// // Recursion function additive components in symmetric case
// // The output vector has dimension 4 and is interpreted as follows
// // output[0] -> positive contribution going to I(S_X) when z is background
// // output[1] -> negative contribution going to I(S_Z) when z is background
// // output[2] -> positive contribution going to I(S_Z) when x is background
// // output[3] -> negative contribution going to I(S_X) when x is background
// vector<double> recurse_additive_sym(int n,
//                                     double* x, double* z, 
//                                     int* I_map,
//                                     int* feature,
//                                     int* child_left,
//                                     int* child_right,
//                                     double* threshold,
//                                     double* value,
//                                     int n_features,
//                                     vector<int> &in_ISX,
//                                     vector<int> &in_ISZ,
//                                     double* result_1,
//                                     double* result_2
//                                     )
// {
//     int current_feature = feature[n];
//     int x_child(0), z_child(0);

//     // Arriving at a Leaf
//     if (child_left[n] < 0)
//     {   
//         vector<double> output(4, 0.0);

//         // First, x is the foreground and z is the background
//         // |I(S_X)| = 1 so the SINGLE element of I(S_X) gets a positive contribution
//         if (in_ISX[n_features]==1){
//             output[0] = value[n];
//         }
//         // |I(S_X)| = 0 so EACH element of I(S_Z) gets a negative contribution
//         else if (in_ISX[n_features]==0){
//             output[1] = -value[n];
//         }

//         // Second, z is the foreground and x is the background
//         // |I(S_Z)| = 1 so the SINGLE element of I(S_Z) gets a positive contribution
//         if (in_ISZ[n_features]==1){
//             output[2] = value[n];
//         }
//         // |I(S_Z)| = 0 so EACH element of I(S_X) gets a negative contribution
//         else if (in_ISZ[n_features]==0){
//             output[3] = -value[n];
//         }
        
//         return output;
//     }

//     // Find children of x and z
//     x_child = x[current_feature] <= threshold[n] ? child_left[n] : child_right[n];
//     z_child = z[current_feature] <= threshold[n] ? child_left[n] : child_right[n];

//     // Scenario 1 : x and z go the same way so we avoid the type B edge
//     if (x_child == z_child){
//         return recurse_additive_sym(x_child, x, z, I_map, feature, child_left, child_right,
//                             threshold, value, n_features, in_ISX, in_ISZ, result_1, result_2);
//     }

//     // Senario 2: x and z go different ways and we have seen this feature I(i) in I(S_X) U I(S_Z).
//     // Hence we go down the correct edge to ensure that I(S_X) and I(S_Z) are kept disjoint
//     if (in_ISX[ I_map[current_feature] ] || in_ISZ[ I_map[current_feature] ]){
//         if (in_ISX[ I_map[current_feature] ]){
//             return recurse_additive_sym(x_child, x, z, I_map, feature, child_left, child_right,
//                             threshold, value, n_features, in_ISX, in_ISZ, result_1, result_2);
//         }
//         else{
//             return recurse_additive_sym(z_child,x, z, I_map, feature, child_left, child_right,
//                             threshold, value, n_features, in_ISX, in_ISZ, result_1, result_2);
//         }
//     }

//     // Scenario 3 : x and z go different ways and we have not yet seen this feature
//     else {
//         vector<double> sum_x_child(4, 0.0);
//         vector<double> sum_z_child(4, 0.0);
//         vector<double> output (4, 0.0);

//         // Go to x's child if it ensures that |I(S_X)|<=1 or |I(S_Z)|<=1
//         if (in_ISX[n_features] == 0 || in_ISZ[n_features] <= 1){
//             in_ISX[ I_map[current_feature] ]++; in_ISX[n_features]++;
//             sum_x_child = recurse_additive_sym(x_child, x, z, I_map, 
//                                                 feature, child_left, child_right,
//                                                 threshold, value, n_features, in_ISX, in_ISZ, 
//                                                 result_1, result_2);
//             in_ISX[ I_map[current_feature] ]--; in_ISX[n_features]--;
//         }

//         // Go to z's child if it ensures that |I(S_X)|<=1 or |I(S_Z)|<=1
//         if (in_ISZ[n_features] == 0 || in_ISX[n_features] <= 1){
//             in_ISZ[ I_map[current_feature] ]++; in_ISZ[n_features]++;
//             sum_z_child = recurse_additive_sym(z_child,x, z, I_map, 
//                                                 feature, child_left, child_right,
//                                                 threshold, value, n_features, in_ISX, in_ISZ,
//                                                 result_1, result_2);
//             in_ISZ[ I_map[current_feature] ]--; in_ISZ[n_features]--;
//         }

//         // Add contribution to the feature
//         // When z is background
//         result_1[ I_map[current_feature] ] += sum_x_child[0] + sum_z_child[1];
//         // When x is background
//         result_2[ I_map[current_feature] ] += sum_x_child[3] + sum_z_child[2];

//         // Add the contribution of both childs and propagate to the parent
//         for (int i(0); i < 4; i++){
//             output[i] = sum_x_child[i] + sum_z_child[i];
//         }
//         return output;
//     }
// }




// Recursion function for Taylor-TreeSHAP
int recurse_taylor_treeshap(int n,
                            double* x, double* z, 
                            struct TreeEnsemble* tree,
                            struct Image* image,
                            vector<int> ISX_U_ISZ,
                            Matrix<double> &W,
                            double* result)
{
    int current_feature = tree->feature[n];
    int x_child(0), z_child(0);
    // num_players := |I(S_{XZ})|
    int num_players = 0;

    // Arriving at a Leaf
    if (tree->left_child[n] < 0)
    {
        num_players = ISX_U_ISZ.size();
        if (num_players == 0){
            return 0;
        }
        for (auto & i : ISX_U_ISZ){
            for (auto & j : ISX_U_ISZ){
                // Diagonal element
                if (i == j) {
                    // i in I(S_Z) and I(S_X) is empty
                    if (image->in_ISZ[i] && (image->card_ISX == 0) ){
                        result[i * image->n_features + i] -= tree->value[n];
                    }
                    // I(S_X) = {i}
                    if (image->in_ISX[i] && (image->card_ISX == 1) ){
                        result[i * image->n_features + i] += tree->value[n];
                    }
                }
                // Non-diagonal element
                else {
                    // i,j in I(S_X)
                    if (image->in_ISX[i] && image->in_ISX[j]){
                        result[i * image->n_features + j] += W[image->card_ISX - 2][num_players-1] * tree->value[n];
                    }
                    // i,j in I(S_Z)
                    else if (image->in_ISZ[i] && image->in_ISZ[j]){
                        result[i * image->n_features + j] += W[image->card_ISX][num_players-1] * tree->value[n];
                    }
                    // i in I(S_X)  and  j in I(S_Z)   OR   j in I(S_X)  and  i in I(S_Z)
                    else if ((image->in_ISX[i] + image->in_ISZ[j] + image->in_ISX[j] + image->in_ISZ[i]) == 2){
                        result[i * image->n_features + j]-= W[image->card_ISX - 1][num_players-1] * tree->value[n];
                    }
                }
            }
        }
        return 0;
    }

    // Find children of x and z
    x_child = x[current_feature] <= tree->threshold[n] ? tree->left_child[n] : tree->right_child[n];
    z_child = z[current_feature] <= tree->threshold[n] ? tree->left_child[n] : tree->right_child[n];

    // Scenario 1 : x and z go the same way so we avoid the type B edge
    if (x_child == z_child){
        return recurse_taylor_treeshap(x_child, x, z, tree, image, ISX_U_ISZ, W, result);
    }

    // Senario 2: x and z go different ways and we have seen this feature I(i) in I(S_X) U I(S_Z).
    // Hence we go down the correct edge to ensure that I(S_X) and I(S_Z) are kept disjoint
    if (image->in_ISX[ image->Imap[current_feature] ] || image->in_ISZ[ image->Imap[current_feature] ]){
        if (image->in_ISX[ image->Imap[current_feature] ]){
            return recurse_taylor_treeshap(x_child, x, z, tree, image, ISX_U_ISZ, W, result);
        }
        else{
            return recurse_taylor_treeshap(z_child, x, z, tree, image, ISX_U_ISZ, W, result);
        }
    }

    // Scenario 3 : x and z go different ways and we have not yet seen this I(i)
    else {
        // Go to x's child
        ISX_U_ISZ.push_back( image->Imap[current_feature] );
        image->in_ISX[ image->Imap[current_feature] ]++; 
        image->card_ISX++;
        recurse_taylor_treeshap(x_child, x, z, tree, image, ISX_U_ISZ, W, result);
        image->in_ISX[ image->Imap[current_feature] ]--;
        image->card_ISX--;

        // Go to z's child
        image->in_ISZ[ image->Imap[current_feature] ]++; 
        image->card_ISZ++;
        recurse_taylor_treeshap(z_child, x, z, tree, image, ISX_U_ISZ, W, result);
        image->in_ISZ[ image->Imap[current_feature] ]--;
        image->card_ISZ--;
        ISX_U_ISZ.pop_back();
        return 0;
    }
}

# endif