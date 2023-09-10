#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// Define a struct for Value, the fundamental building block of the autograd engine
typedef struct Value {
    float val;  // actual value
    float grad;  // gradient
    struct Value** children;  // children this value depends on
    int n_children;  // number of children
    void (*backward)(struct Value*);  // backward function to compute gradients
} Value;

// Initialize a new Value object with a given float
Value* make_value(float x) {
    Value* v = (Value*)malloc(sizeof(Value));
    v->val = x;
    v->grad = 0;
    v->children = NULL;
    v->n_children = 0;
    v->backward = NULL;
    return v;
}

// A simple function to print a Value object
void print_value(Value* v) {
    printf("Value(val=%.2f, grad=%.2f)\\n", v->val, v->grad);
}

// Function to perform backpropagation
// void backward(Value* v) {
//     if (v->backward) {
//         v->backward(v);
//     }
// }

// Helper function for backward to perform topological sort
void build_topo(Value* v, Value** topo, int* topo_size, Value** visited, int* visited_size) {
    for (int i = 0; i < *visited_size; ++i) {
        if (visited[i] == v) return;
    }

    visited[*visited_size] = v;
    (*visited_size)++;
    // printf("%i\n", v->n_children);

    for (int i = 0; i < v->n_children; ++i) {
        printf("child of %f\n", v->val);
        for (int i = 0; i < v->n_children; ++i) {
            print_value(v->children[i]);
        }
        printf("\n\n");
        build_topo(v->children[i], topo, topo_size, visited, visited_size);
    }

    // printf("topo size = %i, node.val = %.2f\n", *topo_size, v->val);
    

    topo[*topo_size] = v;
    (*topo_size)++;

    printf("[ ");
    for (int i = 0; i < *topo_size; ++i) {
        printf("%.2f, ", topo[i]->val);
    }
    printf(" ]\n");
    // printf("topo size = %i, node.val = %.2f\n", *topo_size, v->val);
}

// New backward function
void backward(Value* root) {
    Value* topo[1000];  // Assuming a maximum of 100 nodes in the computation graph for simplicity
    int topo_size = 0;
    Value* visited[1000];
    int visited_size = 0;

    build_topo(root, topo, &topo_size, visited, &visited_size);

    root->grad = 1.0;

    for (int i = topo_size - 1; i >= 0; --i) {
        printf("%.2f", topo[i]->val);
        printf("\n");
        if (topo[i]->backward) {
            topo[i]->backward(topo[i]);
        }
    }
}

// Backward function for addition
void add_backward(Value* v) {
    
    for (int i = 0; i < v->n_children; ++i) {
        // printf("child %.f %.f grad = %f", v->val, v->children[i], v->grad);
        v->children[i]->grad += v->grad;
        // backward(v->children[i]);
    }
}

// Backward function for multiplication
void mul_backward(Value* v) {
    // printf("child %.f grad = %f*%f", v->children[0], v->children[1]->val, v->grad);
    // printf("child %.f grad = %f*%f", v->children[1], v->children[0]->val, v->grad);
    v->children[0]->grad += v->children[1]->val * v->grad;
    v->children[1]->grad += v->children[0]->val * v->grad;
    // backward(v->children[0]);
    // backward(v->children[1]);
}

// Function to perform addition
Value* add(Value* a, Value* b) {
    Value* out = (Value*)malloc(sizeof(Value));
    out->val = a->val + b->val;
    out->grad = 0;
    out->children = (Value**)malloc(2 * sizeof(Value*));
    out->children[0] = a;
    out->children[1] = b;
    out->n_children = 2;
    out->backward = add_backward;
    return out;
}

// Function to perform multiplication
Value* mul(Value* a, Value* b) {
    Value* out = (Value*)malloc(sizeof(Value));
    out->val = a->val * b->val;
    out->grad = 0;
    out->children = (Value**)malloc(2 * sizeof(Value*));
    out->children[0] = a;
    out->children[1] = b;
    out->n_children = 2;
    out->backward = mul_backward;
    return out;
}

// Function to free a Value object
void free_value(Value* v) {
    if (v->children) {
        free(v->children);
    }
    free(v);
}