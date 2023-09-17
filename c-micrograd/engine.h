#include <stdio.h>
#include <stdlib.h>
#include <math.h>

float relu_alpha = 0.01;

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

Value** make_values(float* arr) {
    size_t len = sizeof(arr) / sizeof(arr[0]);
    // Allocate memory for an array of pointers to Value structures
    Value** values = (Value**)malloc(len * sizeof(Value*));
    if (values == NULL) {
        perror("Memory allocation failed");
        exit(1);
    }

    // Initialize each element of the array using the make_value function
    for (size_t i = 0; i < len; i++) {
        values[i] = make_value(arr[i]);
    }
    return values;
}


// A simple function to print a Value object
void print_value(Value* v) {
    printf("Value(val=%.2f, grad=%.2f)\n", v->val, v->grad);
}

// Helper function for backward to perform topological sort
void build_topo(Value* v, Value** topo, int* topo_size, Value** visited, int* visited_size) {
    for (int i = 0; i < *visited_size; ++i) {
        if (visited[i] == v) return;
    }

    visited[*visited_size] = v;
    (*visited_size)++;
    // printf("%i\n", v->n_children);

    for (int i = 0; i < v->n_children; ++i) {
        // printf("child of %f\n", v->val);
        for (int i = 0; i < v->n_children; ++i) {
            // print_value(v->children[i]);
        }
        // printf("\n\n");
        build_topo(v->children[i], topo, topo_size, visited, visited_size);
    }

    // printf("topo size = %i, node.val = %.2f\n", *topo_size, v->val);
    

    topo[*topo_size] = v;
    (*topo_size)++;

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
        // printf("%.2f", topo[i]->val);
        // printf("\n");
        if (topo[i]->backward) {
            topo[i]->backward(topo[i]);
        }
    }
}

void grad_clip(Value* v, float min_val, float max_val) {
    if (v->grad < min_val) {
        v->grad = min_val;
    } else if (v->grad > max_val) {
        v->grad = max_val;
    }
}

// Backward function for addition
void add_backward(Value* v) {
    
    v->children[0]->grad += v->grad;
    v->children[1]->grad += v->grad;
    grad_clip(v->children[0], -10.0, 10.0);
    grad_clip(v->children[1], -10.0, 10.0);
}

// Backward function for multiplication
void mul_backward(Value* v) {
    // printf("child %.f grad = %f*%f", v->children[0], v->children[1]->val, v->grad);
    // printf("child %.f grad = %f*%f", v->children[1], v->children[0]->val, v->grad);
    v->children[0]->grad += v->children[1]->val * v->grad;
    v->children[1]->grad += v->children[0]->val * v->grad;
    grad_clip(v->children[0], -10.0, 10.0);
    grad_clip(v->children[1], -10.0, 10.0);
}

// Backward function for division
void div_backward(Value* v) {
    v->children[0]->grad += (1.0 / v->children[1]->val) * v->grad;
    v->children[1]->grad += (-v->children[0]->val / (v->children[1]->val * v->children[1]->val)) * v->grad;
    grad_clip(v->children[0], -10.0, 10.0);
    grad_clip(v->children[1], -10.0, 10.0);
}

// Backward function for power
void power_backward(Value* v) {
    v->children[0]->grad += (v->children[1]->val * pow(v->children[0]->val, v->children[1]->val - 1)) * v->grad;
    if (v->children[0]->val > 0) {  // Ensure base is positive before computing log
        v->children[1]->grad += (log(v->children[0]->val) * pow(v->children[0]->val, v->children[1]->val)) * v->grad;
    }
    grad_clip(v->children[0], -10.0, 10.0);
    grad_clip(v->children[1], -10.0, 10.0);
}

// Backward function for subtraction
void sub_backward(Value* v) {
    v->children[0]->grad += v->grad;
    v->children[1]->grad -= v->grad;
    grad_clip(v->children[0], -10.0, 10.0);
    grad_clip(v->children[1], -10.0, 10.0);
}

// Backward function for Leaky ReLU
void leaky_relu_backward(Value* v) {
    if (v->children[0]->val > 0) {
        v->children[0]->grad += v->grad;
    } else {
        v->children[0]->grad += v->grad * relu_alpha;
    }
    grad_clip(v->children[0], -10.0, 10.0);
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

// Function to perform division
Value* divide(Value* a, Value* b) {
    if(b->val == 0.0) {
        printf("Error: Division by zero\n");
        exit(1);  // Or handle the error in another way
    }

    Value* out = (Value*)malloc(sizeof(Value));
    out->val = a->val / b->val;
    out->grad = 0;
    out->children = (Value**)malloc(2 * sizeof(Value*));
    out->children[0] = a;
    out->children[1] = b;
    out->n_children = 2;
    out->backward = div_backward;
    return out;
}

// Function to perform power
Value* power(Value* a, Value* b) {
    Value* out = (Value*)malloc(sizeof(Value));
    out->val = pow(a->val, b->val);
    out->grad = 0;
    out->children = (Value**)malloc(2 * sizeof(Value*));
    out->children[0] = a;
    out->children[1] = b;
    out->n_children = 2;
    out->backward = power_backward;
    return out;
}

// Function to perform subtraction
Value* sub(Value* a, Value* b) {
    Value* out = (Value*)malloc(sizeof(Value));
    out->val = a->val - b->val;
    out->grad = 0;
    out->children = (Value**)malloc(2 * sizeof(Value*));
    out->children[0] = a;
    out->children[1] = b;
    out->n_children = 2;
    out->backward = sub_backward;
    return out;
}

// Forward function for Leaky ReLU
Value* leaky_relu(Value* a) {
    Value* out = (Value*)malloc(sizeof(Value));

    if (a->val > 0) {
        out->val = a->val;
    } else {
        out->val = relu_alpha * a->val;
    }

    out->grad = 0;
    out->children = (Value**)malloc(sizeof(Value*));
    out->children[0] = a;
    out->n_children = 1;
    out->backward = leaky_relu_backward;

    return out;
}

// TODO SOFTMAX
// Backward function for Softmax
// void softmax_backward(Value* v) {
//     for (int i = 0; i < v->n_children; i++) {
//         Value* child = v->children[i];
//         for (int j = 0; j < v->n_children; j++) {
//             if (i == j) {
//                 child->grad += v->grad * v->val * (1 - v->val);  // When i == j
//             } else {
//                 child->grad -= v->grad * v->val * v->children[j]->val;  // When i != j
//             }
//             grad_clip(child, -10.0, 10.0);
//         }
//     }
// }

// // Forward function for Softmax
// Value** softmax(Value** x, int size) {
//     Value** out = (Value**)malloc(size * sizeof(Value*));
//     double sum_exp = 0.0;

//     // Calculate the sum of exponentials
//     for (int i = 0; i < size; i++) {
//         sum_exp += exp(x[i]->val);
//     }

//     // Compute the softmax values
//     for (int i = 0; i < size; i++) {
//         out[i] = (Value*)malloc(sizeof(Value));
//         out[i]->val = exp(x[i]->val) / sum_exp;
//         out[i]->grad = 0;
//         out[i]->children = (Value**)malloc(size * sizeof(Value*));
//         for (int j = 0; j < size; j++) {
//             out[i]->children[j] = x[j];
//         }
//         out[i]->n_children = size;
//         out[i]->backward = softmax_backward;
//     }
//     return out;
// }




// Function to free a Value object
void free_value(Value* v) {
    if (v->children) {
        free(v->children);
    }
    free(v);
}