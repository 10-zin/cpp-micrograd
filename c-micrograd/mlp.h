#include "engine.h"
#include <time.h>

typedef struct Neuron {
    Value** w;  // array of weights
    Value* b;   // bias
    int nin;    // number of input neurons
    int nonlin; // nonlinearity flag: 1 for ReLU, 0 for linear
} Neuron;

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
        Value* sum = leaky_relu(sum);
    }

    return sum;
}

typedef struct Layer {
    Neuron** neurons;  // array of neurons
    int nout;          // number of output neurons
} Layer;

Layer* init_layer(int nin, int nout, int nonlin) {
    Layer* layer = (Layer*)malloc(sizeof(Layer));
    layer->neurons = (Neuron**)malloc(nout * sizeof(Neuron*));
    for (int i = 0; i < nout; i++) {
        
        layer->neurons[i] = init_neuron(nin, nonlin);
    }
    layer->nout = nout;
    return layer;
}

Value** layer_forward(Layer* layer, Value** x) {
    Value** out = (Value**)malloc(layer->nout * sizeof(Value*));
    for (int i = 0; i < layer->nout; i++) {
        out[i] = neuron_forward(layer->neurons[i], x);
    }
    return out;
}

typedef struct MLP {
    Layer** layers;  // array of layers
    int nlayers;     // number of layers
} MLP;

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

Value** mlp_forward(MLP* mlp, Value** x) {
    for (int i = 0; i < mlp->nlayers; i++) {
        x = layer_forward(mlp->layers[i], x);
    }
    return x;
}

// Mean Squared Error loss
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

void update_weights(Value* v, float lr) {
    v->val -= lr * v->grad;
}

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

void free_neuron(Neuron* neuron) {
    for (int i = 0; i < neuron->nin; i++) {
        free_value(neuron->w[i]);
    }
    free(neuron->w);
    free_value(neuron->b);
    free(neuron);
}

void free_layer(Layer* layer) {
    for (int i = 0; i < layer->nout; i++) {
        free_neuron(layer->neurons[i]);
    }
    free(layer->neurons);
    free(layer);
}

void free_mlp(MLP* mlp) {
    for (int i = 0; i < mlp->nlayers; i++) {
        free_layer(mlp->layers[i]);
    }
    free(mlp->layers);
    free(mlp);
}

