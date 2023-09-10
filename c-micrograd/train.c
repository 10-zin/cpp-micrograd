#include "mlp.h"
#include "load.h"

int main() {
    srand(43);  // seed the random number generator

    // Create a custom MLP with sizes [2,10,5,2]
    int sizes[] = {1, 5, 10 5, 2};
    int nlayers = sizeof(sizes) / sizeof(int);
    printf("nlayers:%i\n", nlayers);
    MLP* mlp = init_mlp(sizes, nlayers);

    // Create a dummy training input and label
    Value* x[2];
    x[0] = make_value(0.5);
    x[1] = make_value(-0.5);
    Value* y_true[2];
    y_true[0] = make_value(1);
    y_true[1] = make_value(0);
    Entry* entries = load_data();

    // Train for a few iterations
    int epochs = 2;
    float lr = 0.001;
    for (int i = 0; i < epochs; i++) {
        for (int i =0; i < MAX_ENTRIES; i++) {
            float arr_x[] = {entries[i].number};
            float arr_y[] = {entries[i].label};
            Value** x = make_values(arr_x);
            Value** y_true = make_values(arr_y);
            train(mlp, x, y_true, lr);
        }
        
    }

    for (int i = 0; i < mlp->nlayers; i++) {
        Layer* layer = mlp->layers[i];
        for (int j = 0; j < layer->nout; j++) {
            Neuron* neuron = layer->neurons[j];
            // update_weights(neuron->b, lr);
            for (int k = 0; k < neuron->nin; k++) {
                // update_weights(neuron->w[k], lr);
                print_value(neuron->w[k]);
            }
        }
    }

    free_mlp(mlp);
    free_value(x[0]);
    free_value(x[1]);
    free_value(y_true[0]);
    free_value(y_true[1]);

    return 0;
}
