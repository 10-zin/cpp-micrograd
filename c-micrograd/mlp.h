#include "engine.h"
#include <time.h>


/**
 * @struct Neuron
 * @brief Represents a single neuron in a neural network layer.
 *
 * A neuron has weights corresponding to each input, a bias term, and an activation function.
 */

typedef struct Neuron {
    Value** w;  // array of weights
    Value* b;   // bias
    int nin;    // number of input neurons
    int nonlin; // nonlinearity flag: 1 for ReLU, 0 for linear
} Neuron;

/**
 * @brief Initialize a neuron with random weights and zero bias.
 *
 * @param nin Number of input connections.
 * @param nonlin Activation function flag (1 for ReLU, 0 for linear).
 * @return Pointer to the initialized Neuron.
 *
 * @example
 * Neuron* my_neuron = init_neuron(3, 1);  // A neuron with 3 inputs and ReLU activation
 */

Neuron* init_neuron(int nin, int nonlin) {
    Neuron* neuron = (Neuron*)malloc(sizeof(Neuron));
    neuron->w = (Value**)malloc(nin * sizeof(Value*));
    for (int i = 0; i < nin; i++) {
        neuron->w[i] = make_value((rand() % 2000 - 1000) / 1000.0);  // random values between -1 and 1
    }
    neuron->b = make_value(0);
    neuron->nin = nin;
    neuron->nonlin = nonlin;
    return neuron;
}


/**
 * @brief Perform forward pass computation for a neuron.
 *
 * @param neuron Pointer to the neuron.
 * @param x Array of input values.
 * @return Pointer to the output Value.
 *
 * @example
 * Value* input_values[3] = {make_value(1.0), make_value(0.5), make_value(-0.5)};
 * Value* output = neuron_forward(my_neuron, input_values);
 */
Value* neuron_forward(Neuron* neuron, Value** x) {
    Value* sum = make_value(0);
    for (int i = 0; i < neuron->nin; i++) {
        Value* prod = mul(neuron->w[i], x[i]);
        // printf("\nw%f*x%f, %f", neuron->w[i]->val, x[i]->val, prod->val);
        sum = add(sum, prod);
        // free_value(prod);
    }
    sum = add(sum, neuron->b);
    if (neuron->nonlin) {
        sum = leaky_relu(sum);
    }

    return sum;
}

/**
 * @struct Layer
 * @brief Represents a single layer in the neural network.
 *
 * A layer consists of multiple neurons.
 */
typedef struct Layer {
    Neuron** neurons;  // array of neurons
    int nout;          // number of output neurons
} Layer;

/**
 * @brief Initialize a neural network layer with specified neurons.
 *
 * @param nin Number of input connections for each neuron.
 * @param nout Number of neurons in the layer.
 * @param nonlin Activation function flag for all neurons (1 for ReLU, 0 for linear).
 * @return Pointer to the initialized Layer.
 *
 * @example
 * Layer* my_layer = init_layer(3, 2, 1);  // A layer with 2 neurons, each having 3 inputs and ReLU activation
 */
Layer* init_layer(int nin, int nout, int nonlin) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    layer->neurons = (Neuron**)malloc(nout * sizeof(Neuron*));
    for (int i = 0; i < nout; i++) {
        
        layer->neurons[i] = init_neuron(nin, nonlin);
    }
    layer->nout = nout;
    return layer;
}

/**
 * @brief Perform forward pass computation for a layer.
 *
 * @param layer Pointer to the layer.
 * @param x Array of input values for the layer.
 * @return Array of output values from all neurons in the layer.
 *
 * @example
 * Value* input_values[3] = {make_value(1.0), make_value(0.5), make_value(-0.5)};
 * Value** outputs = layer_forward(my_layer, input_values);
 */
Value** layer_forward(Layer* layer, Value** x) {
    Value** out = (Value**)malloc(layer->nout * sizeof(Value*));
    for (int i = 0; i < layer->nout; i++) {
        out[i] = neuron_forward(layer->neurons[i], x);
    }
    return out;
}

/**
 * @struct MLP
 * @brief Represents a Multilayer Perceptron (MLP) neural network.
 *
 * An MLP consists of multiple layers.
 */
typedef struct MLP {
    Layer** layers;  // array of layers
    int nlayers;     // number of layers
} MLP;

/**
 * @brief Initialize a Multilayer Perceptron (MLP) with the specified layer sizes.
 *
 * @param sizes Array of layer sizes, where each element represents the number of neurons in that layer.
 * @param nlayers Number of layers in the MLP.
 * @return Pointer to the initialized MLP.
 *
 * @example
 * int layer_sizes[3] = {3, 4, 2};
 * MLP* my_mlp = init_mlp(layer_sizes, 3);  // An MLP with 3 layers: 3 neurons, 4 neurons, and 2 neurons respectively
 */
MLP* init_mlp(int* sizes, int nlayers) {
    MLP* mlp = (MLP*)malloc(sizeof(MLP));
    mlp->layers = (Layer**)malloc((nlayers - 1) * sizeof(Layer*));
    for (int i = 0; i < nlayers - 1; i++) {
        int nonlin = (i != nlayers - 2);  // nonlinearity for all layers except the last one
        mlp->layers[i] = init_layer(sizes[i], sizes[i+1], nonlin);
    }
    mlp->nlayers = nlayers - 1;
    return mlp;
}

/**
 * @brief Perform forward pass computation for the entire MLP.
 *
 * @param mlp Pointer to the MLP.
 * @param x Array of input values for the MLP.
 * @return Array of output values from the final layer of the MLP.
 *
 * @example
 * Value* input_values[3] = {make_value(1.0), make_value(0.5), make_value(-0.5)};
 * Value** outputs = mlp_forward(my_mlp, input_values);
 */
Value** mlp_forward(MLP* mlp, Value** x) {
    for (int i = 0; i < mlp->nlayers; i++) {
        x = layer_forward(mlp->layers[i], x);
    }
    return x;
}

/**
 * @brief Compute the mean squared error (MSE) loss between predicted and true values.
 *
 * @param y_pred Array of predicted values.
 * @param y_true Array of true values.
 * @param size Number of values in y_pred and y_true arrays.
 * @return Pointer to the computed MSE loss value.
 *
 * @example
 * Value* predicted[2] = {make_value(0.9), make_value(0.1)};
 * Value* true_values[2] = {make_value(1.0), make_value(0.0)};
 * Value* loss = mse_loss(predicted, true_values, 2);
 */
Value* mse_loss(Value** y_pred, Value** y_true, int size) {
    
    Value* loss = make_value(0.0);
    for (int i = 0; i < size; i++) {
        Value* diff = sub(y_pred[i], y_true[i]);
        Value* sq = power(diff, make_value(2.0));
        loss = add(loss, sq);
    }
    loss = divide(loss, make_value(size));

    return loss;
}

/**
 * @brief Update the weights of a value using gradient descent.
 *
 * @param v Pointer to the value whose weights need to be updated.
 * @param lr Learning rate for the weight update.
 *
 * @example
 * Value* weight = make_value(0.5);
 * weight->grad = -0.1;  // Example gradient
 * update_weights(weight, 0.01);  // Updates weight using gradient descent
 */
void update_weights(Value* v, float lr) {
    v->val -= lr * v->grad;
}

/**
 * @brief Display the parameters (weights and biases) of the MLP.
 *
 * @param mlp Pointer to the MLP.
 *
 * @example
 * show_params(my_mlp);  // Prints the weights and biases of all layers and neurons in the MLP
 */
void show_params(MLP* mlp){
    printf("\nMLP\n");
    for (int i = 0; i < mlp->nlayers; i++) {
        Layer* layer = mlp->layers[i];
        printf("\nLayer%i:\n", i);
        for (int j = 0; j < layer->nout; j++) {
            Neuron* neuron = layer->neurons[j];
            for (int k = 0; k < neuron->nin; k++) {
                print_value(neuron->w[k]);
            }
        }
    }
        printf("\n\n");
}

/**
 * @brief Train the MLP for a single input-output pair and update its weights using gradient descent.
 *
 * @param mlp Pointer to the MLP.
 * @param x Array of input values for training.
 * @param y_true Array of true output values for training.
 * @param lr Learning rate for weight updates.
 * @return Pointer to the computed loss value for the input-output pair.
 *
 * @example
 * Value* input_values[3] = {make_value(1.0), make_value(0.5), make_value(-0.5)};
 * Value* true_output[2] = {make_value(0.9), make_value(0.1)};
 * Value* loss = train(my_mlp, input_values, true_output, 0.01);
 */
Value* train(MLP* mlp, Value** x, Value** y_true, float lr) {

    // Forward pass
    Value** y_pred = mlp_forward(mlp, x);

    // Compute loss
    Value* loss = mse_loss(y_pred, y_true, 2);
    // printf("Loss: %.2f\n", loss->val);

    // Update weights and biases using gradient descent
    for (int i = 0; i < mlp->nlayers; i++) {
        Layer* layer = mlp->layers[i];
        for (int j = 0; j < layer->nout; j++) {
            Neuron* neuron = layer->neurons[j];
            update_weights(neuron->b, lr);
            for (int k = 0; k < neuron->nin; k++) {
                update_weights(neuron->w[k], lr);
            }
        }
    }

    return loss;

    // free_value(loss);
}

/**
 * @brief Free the memory allocated for a neuron.
 *
 * @param neuron Pointer to the neuron to be freed.
 */
void free_neuron(Neuron* neuron) {
    for (int i = 0; i < neuron->nin; i++) {
        free_value(neuron->w[i]);
    }
    free(neuron->w);
    free_value(neuron->b);
    free(neuron);
}

/**
 * @brief Free the memory allocated for a layer.
 *
 * @param layer Pointer to the layer to be freed.
 */
void free_layer(Layer* layer) {
    for (int i = 0; i < layer->nout; i++) {
        free_neuron(layer->neurons[i]);
    }
    free(layer->neurons);
    free(layer);
}

/**
 * @brief Free the memory allocated for the entire MLP.
 *
 * @param mlp Pointer to the MLP to be freed.
 */
void free_mlp(MLP* mlp) {
    for (int i = 0; i < mlp->nlayers; i++) {
        free_layer(mlp->layers[i]);
    }
    free(mlp->layers);
    free(mlp);
}

