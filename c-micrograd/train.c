#include "mlp.h"
#include "load.h"

// One-hot encoding of label. mlp will predict a (2,) dimensional vector, for classification.
// So if label is 0 -> [1, 0], if 1 -> [0, 1]
// pred[0]<pred[1] -> mlp predicted 0 (odd)
// pred[1]<pred[0] -> mlp predicted 1 (even)
float* one_hot_encode(float y_true) {
    float* encoded = (float*)malloc(2 * sizeof(float));
    if (y_true==1.0){
        encoded[0] = 0.0;
        encoded[1] = 1.0;
    }
    else{
        encoded[0] = 1.0;
        encoded[1] = 0.0;
    }
    return encoded;
}

int main() {
    srand(43);  // seed the random number generator

    // labels, in this case 2, (odd (0) or even (1))
    int labels=2;
    // custom MLP with sizes [1, 5, 10, 5, 2]
    int sizes[] = {1, 5, 10, 5, labels};
    int nlayers = sizeof(sizes) / sizeof(int);
    
    // backprop after seeing every 2 data instances
    int backward_freq = 2;

    // Init a MLP with custom layer sizes.
    MLP* mlp = init_mlp(sizes, nlayers);

    // load data from data.txt
    Entry* entries = load_data();

    // Train for a n epochs.
    int epochs = 50;

    float lr = 0.001;
    Value* total_loss = make_value(0.0);
    float epoch_loss = 0.0;

    // show_params(mlp);
    for (int ep = 0; ep < epochs; ep++) {
        for (int i=0; i < 25; i++) {
            
            float arr_x[] = {entries[i].number};
            Value** x = make_values(arr_x);

            float* arr_y = one_hot_encode(entries[i].label);
            Value** y_true = make_values(arr_y);

            Value* loss = train(mlp, x, y_true, lr);
            total_loss = add(total_loss, loss);
            epoch_loss+=total_loss->val;

            // Backward pass
            if (i%backward_freq==0){
                // make loss.grad=0 (last node in mlp topo graph).
                // grad is basically dy/da or dy/db where y = a op b; op can by anything add, sub, div ..
                // in case of last node (which is the loss) -> a or b is itself y. since its the last node, and does have any op on it.
                // so basically dy/dy = 1. 
                // This kicks off the gradient propogation backwards.
                total_loss->grad=1.0;
                backward(total_loss);
                // resetting total_loss for a new epoch.
                total_loss = make_value(0.0);
            }
            
            
            // show_params(mlp);
        }
        printf("\n\nEPOCH LOSSS!!::%f: \n\n", epoch_loss/25);
        epoch_loss=0.0;
        
    }

    free_mlp(mlp);

    return 0;
}
