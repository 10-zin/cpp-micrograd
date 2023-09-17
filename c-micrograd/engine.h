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

/**
 * @brief Initialize a new Value object with a given float.
 *
 * This function allocates memory for a Value object and initializes its attributes.
 *
 * @param x The float value to initialize the Value object with.
 * @return A pointer to the newly created Value object.
 *
 * @example
 * Value* v = make_value(5.0);
 * print_value(v);  // Outputs: Value(val=5.00, grad=0.00)
 */
Value* make_value(float x) {
    Value* v = (Value*)malloc(sizeof(Value));
    v->val = x;
    v->grad = 0;
    v->children = NULL;
    v->n_children = 0;
    v->backward = NULL;
    return v;
}

/**
 * @brief Creates an array of Value objects from a float array.
 *
 * This function allocates memory for an array of Value objects and initializes each element using the given float array.
 *
 * @param arr Pointer to the float array.
 * @return A pointer to an array of Value objects.
 *
 * @example
 * float arr[] = {1.0, 2.0, 3.0};
 * Value** values = make_values(arr);
 * for (int i = 0; i < 3; i++) {
 *     print_value(values[i]);  // Outputs: Value(val=1.00, grad=0.00), Value(val=2.00, grad=0.00), etc.
 * }
 */
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


/**
 * @brief Prints the attributes of a Value object.
 *
 * This function outputs the 'val' and 'grad' attributes of the given Value object to the console.
 *
 * @param v Pointer to the Value object to be printed.
 *
 * @example
 * Value* v = make_value(3.0);
 * print_value(v);  // Outputs: Value(val=3.00, grad=0.00)
 */
void print_value(Value* v) {
    printf("Value(val=%.2f, grad=%.2f)\n", v->val, v->grad);
}

/**
 * @brief Helper function for backward propagation using topological sort.
 *
 * This function builds a topological order of the computation graph, starting from the given Value object.
 *
 * @param v The starting Value object for the topological sort.
 * @param topo A pointer to an array where the topological order will be stored.
 * @param topo_size Pointer to the size of the topo array.
 * @param visited Pointer to an array that keeps track of visited Value objects.
 * @param visited_size Pointer to the size of the visited array.
 */
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

/**
 * @brief Compute the backward pass to calculate gradients.
 *
 * This function traverses the computation graph in topological order to compute gradients for each Value object.
 *
 * @param v The starting Value object for the backward pass.
 */
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

/**
 * @brief Clip the gradient of a Value object within a specified range.
 *
 * This function ensures that the gradient of the given Value object does not exceed the provided minimum and maximum values.
 *
 * @param v Pointer to the Value object whose gradient needs to be clipped.
 * @param min Minimum allowed value for the gradient.
 * @param max Maximum allowed value for the gradient.
 *
 * @example
 * Value* v = make_value(5.0);
 * v->grad = 15.0;
 * grad_clip(v, 0.0, 10.0);
 * print_value(v);  // Outputs: Value(val=5.00, grad=10.00)
 */
void grad_clip(Value* v, float min_val, float max_val) {
    if (v->grad < min_val) {
        v->grad = min_val;
    } else if (v->grad > max_val) {
        v->grad = max_val;
    }
}


void add_backward(Value* v) {
    
    v->children[0]->grad += v->grad;
    v->children[1]->grad += v->grad;
    grad_clip(v->children[0], -10.0, 10.0);
    grad_clip(v->children[1], -10.0, 10.0);
}

/**
 * @brief Backward function for multiplication operation.
 *
 * Computes the gradient of the multiplication operation with respect to its operands.
 *
 * @param v Pointer to the Value object resulting from the multiplication.
 *
 * @note
 * The final gradient for the operand is its local gradient multiplied by any external gradient flowing from a parent.
 * The local derivative for the multiplication is:
 *     dv/da (locally) = b
 *     dv/db (locally) = a
 * The external gradient (from parent nodes) is stored in v->grad.
 * Thus, the final gradient for a is: dv/da = b * v->grad
 * And for b is: dv/db = a * v->grad
 */
void mul_backward(Value* v) {
    // printf("child %.f grad = %f*%f", v->children[0], v->children[1]->val, v->grad);
    // printf("child %.f grad = %f*%f", v->children[1], v->children[0]->val, v->grad);
    v->children[0]->grad += v->children[1]->val * v->grad;
    v->children[1]->grad += v->children[0]->val * v->grad;
    grad_clip(v->children[0], -10.0, 10.0);
    grad_clip(v->children[1], -10.0, 10.0);
}

/**
 * @brief Backward function for division operation.
 *
 * Computes the gradient of the division operation with respect to its operands.
 *
 * @param v Pointer to the Value object resulting from the division.
 *
 * @note
 * The final gradient for the operand is its local gradient multiplied by any external gradient flowing from a parent.
 * The local derivative for the division is:
 *     dv/da (locally) = 1/b
 *     dv/db (locally) = -a/(b^2)
 * The external gradient (from parent nodes) is stored in v->grad.
 * Thus, the final gradient for a is: dv/da = (1/b) * v->grad
 * And for b is: dv/db = (-a/(b^2)) * v->grad
 */
void div_backward(Value* v) {
    v->children[0]->grad += (1.0 / v->children[1]->val) * v->grad;
    v->children[1]->grad += (-v->children[0]->val / (v->children[1]->val * v->children[1]->val)) * v->grad;
    grad_clip(v->children[0], -10.0, 10.0);
    grad_clip(v->children[1], -10.0, 10.0);
}

/**
 * @brief Backward function for power operation.
 *
 * Computes the gradient of the power operation with respect to its operands.
 *
 * @param v Pointer to the Value object resulting from the power operation.
 *
 * @note
 * The final gradient for the operand is its local gradient multiplied by any external gradient flowing from a parent.
 * The local derivative for the power operation is:
 *     dv/da (locally) = b * a^(b-1)
 *     dv/db (locally) = a^b * log(a)
 * The external gradient (from parent nodes) is stored in v->grad.
 * Thus, the final gradient for a is: dv/da = (b * a^(b-1)) * v->grad
 * And for b is: dv/db = (v * log(a)) * v->grad
 */
void power_backward(Value* v) {
    v->children[0]->grad += (v->children[1]->val * pow(v->children[0]->val, v->children[1]->val - 1)) * v->grad;
    if (v->children[0]->val > 0) {  // Ensure base is positive before computing log
        v->children[1]->grad += (log(v->children[0]->val) * pow(v->children[0]->val, v->children[1]->val)) * v->grad;
    }
    grad_clip(v->children[0], -10.0, 10.0);
    grad_clip(v->children[1], -10.0, 10.0);
}


/**
 * @brief Backward function for subtraction operation.
 *
 * Computes the gradient of the subtraction operation with respect to its operands.
 *
 * @param v Pointer to the Value object resulting from the subtraction.
 *
 * @note
 * The final gradient for the operand is its local gradient multiplied by any external gradient flowing from a parent.
 * The local derivative for the subtraction is:
 *     dv/da (locally) = 1
 *     dv/db (locally) = -1
 * The external gradient (from parent nodes) is stored in v->grad.
 * Thus, the final gradient for a is: dv/da = 1 * v->grad
 * And for b is: dv/db = -1 * v->grad
 */
void sub_backward(Value* v) {
    v->children[0]->grad += v->grad;
    v->children[1]->grad -= v->grad;
    grad_clip(v->children[0], -10.0, 10.0);
    grad_clip(v->children[1], -10.0, 10.0);
}


/**
 * @brief Backward function for Leaky ReLU activation.
 *
 * Computes the gradient of the Leaky ReLU operation with respect to its input.
 *
 * @param v Pointer to the Value object resulting from the Leaky ReLU activation.
 *
 * @note
 * The final gradient for the operand is its local gradient multiplied by any external gradient flowing from a parent.
 * The local derivative for the Leaky ReLU is:
 *     dv/da (locally) = 1 if a > 0
 *     dv/da (locally) = relu_alpha otherwise
 * The external gradient (from parent nodes) is stored in v->grad.
 * Thus, the final gradient for a is: dv/da = (chosen local derivative) * v->grad
 */
void leaky_relu_backward(Value* v) {
    if (v->children[0]->val > 0) {
        v->children[0]->grad += v->grad;
    } else {
        v->children[0]->grad += v->grad * relu_alpha;
    }
    grad_clip(v->children[0], -10.0, 10.0);
}


/**
 * @brief Forward function for addition operation.
 *
 * This function creates a new Value object that represents the sum of two given Value objects.
 * The resulting Value object will have a backward function assigned for gradient computation.
 *
 * @param a Pointer to the first Value object.
 * @param b Pointer to the second Value object.
 * @return A pointer to the new Value object representing the sum.
 *
 * @example
 * Value* v1 = make_value(3.0);
 * Value* v2 = make_value(4.0);
 * Value* sum_val = add(v1, v2);
 * print_value(sum_val);  // Outputs: Value(val=7.00, grad=0.00)
 */
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

/**
 * @brief Forward function for multiplication operation.
 *
 * This function creates a new Value object that represents the product of two given Value objects.
 * The resulting Value object will have a backward function assigned for gradient computation.
 *
 * @param a Pointer to the first Value object.
 * @param b Pointer to the second Value object.
 * @return A pointer to the new Value object representing the product.
 *
 * @example
 * Value* v1 = make_value(3.0);
 * Value* v2 = make_value(4.0);
 * Value* product_val = mul(v1, v2);
 * print_value(product_val);  // Outputs: Value(val=12.00, grad=0.00)
 */
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

/**
 * @brief Forward function for division operation.
 *
 * This function creates a new Value object that represents the quotient of two given Value objects.
 * The resulting Value object will have a backward function assigned for gradient computation.
 *
 * @param a Pointer to the numerator Value object.
 * @param b Pointer to the denominator Value object.
 * @return A pointer to the new Value object representing the quotient.
 *
 * @example
 * Value* v1 = make_value(8.0);
 * Value* v2 = make_value(4.0);
 * Value* quotient_val = divide(v1, v2);
 * print_value(quotient_val);  // Outputs: Value(val=2.00, grad=0.00)
 */
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

/**
 * @brief Forward function for power operation.
 *
 * This function creates a new Value object that represents one Value object raised to the power of another.
 * The resulting Value object will have a backward function assigned for gradient computation.
 *
 * @param base Pointer to the base Value object.
 * @param exponent Pointer to the exponent Value object.
 * @return A pointer to the new Value object representing the power result.
 *
 * @example
 * Value* v1 = make_value(2.0);
 * Value* v2 = make_value(3.0);
 * Value* power_val = power(v1, v2);
 * print_value(power_val);  // Outputs: Value(val=8.00, grad=0.00)
 */
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

/**
 * @brief Forward function for subtraction operation.
 *
 * This function creates a new Value object that represents the difference between two given Value objects.
 * The resulting Value object will have a backward function assigned for gradient computation.
 *
 * @param a Pointer to the minuend Value object.
 * @param b Pointer to the subtrahend Value object.
 * @return A pointer to the new Value object representing the difference.
 *
 * @example
 * Value* v1 = make_value(7.0);
 * Value* v2 = make_value(4.0);
 * Value* diff_val = sub(v1, v2);
 * print_value(diff_val);  // Outputs: Value(val=3.00, grad=0.00)
 */
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

/**
 * @brief Forward function for Leaky ReLU activation.
 *
 * This function creates a new Value object that represents the Leaky ReLU activation of a given Value object.
 * The resulting Value object will have a backward function assigned for gradient computation.
 *
 * @param a Pointer to the input Value object.
 * @return A pointer to the new Value object representing the Leaky ReLU activation.
 *
 * @example
 * Value* v = make_value(-0.5);
 * Value* activated_val = leaky_relu(v);
 * print_value(activated_val);  // Outputs might be: Value(val=-0.005, grad=0.00) depending on relu_alpha value.
 */
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

/**
 * @brief Function to deallocate memory for a Value object.
 *
 * This function frees the memory allocated for a Value object and its children.
 *
 * @param v Pointer to the Value object to be deallocated.
 */
void free_value(Value* v) {
    if (v->children) {
        free(v->children);
    }
    free(v);
}