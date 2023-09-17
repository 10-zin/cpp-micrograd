#include "mlp.h"
#include "load.h"


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

    // Create a custom MLP with sizes [2,10,5,2]
    int labels=2;
    int sizes[] = {1, 5, 10, 5, labels};
    int nlayers = sizeof(sizes) / sizeof(int);
    int backward_freq = 2;
    printf("nlayers:%i\n", nlayers);
    MLP* mlp = init_mlp(sizes, nlayers);

    Entry* entries = load_data();

    // Train for a few iterations
    int epochs = 50;
    float lr = 0.001;
    Value* total_loss = make_value(0.0);
    float epoch_loss = 0.0;
    // show_params(mlp);
    for (int ep = 0; ep < epochs; ep++) {
        for (int i =0; i < 25; i++) {
            
            float arr_x[] = {entries[i].number};
            Value** x = make_values(arr_x);

            float* arr_y = one_hot_encode(entries[i].label);
            Value** y_true = make_values(arr_y);

            Value* loss = train(mlp, x, y_true, lr);
            total_loss = add(total_loss, loss);
            epoch_loss+=total_loss->val;

            // Backward pass
            if (i%backward_freq==0){
                total_loss->grad=1.0;
                backward(total_loss);
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
