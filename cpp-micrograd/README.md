## CPP Micrograd

### Getting Started
1. To play with the autogrand engine, run the following.
    ```
    > g++ engine.cpp playground.cpp -o autograd
    > ./autograd
    ```
2. You can edit playgound.cpp to try other combinations of operations.

### Sample Training script
1. `train.cpp` is a simple script to train a neural net to model the `AND logic gate`.
2. Complile and run it like this:
    ```
    > g++ engine.cpp nn.cpp train.cpp -o train
    > ./train
    ```
3. It will show the MLP architecture, weights for each neuron upon initialization.
4. Then it will create a training set of `AND logic gate`, in a random fashion.
    ```
    Expected behaviour:
        Input  | output
         0 0   | 0
         0 1   | 0
         1 0   | 0
         1 1   | 1
    ```
5. the mlp architecture follows a classification formulation
    1. that is, in the final layer, there are 2 neurons,
    2. first neuron represents value -> 0, 2nd represents 1.
    3. whichever neuron has higher value, is taken as the predicted value by model.
6. Then a trainin loop is done, and loss is calulcated via a simple mean squared error.
7. Iteration loop something like this can be seen. Observe how the loss keeps reducing, indicating that the model is in-fact learning.
    ```Training loop:
            Iteration 0 Loss: -0.476241
            Iteration 1 Loss: -1.86725
            Iteration 2 Loss: -1.0974
            Iteration 3 Loss: -1.51375
            Iteration 4 Loss: -3.9554
            Iteration 5 Loss: -3.95395
            Iteration 6 Loss: -4.40879
            Iteration 7 Loss: -13.7386
            Iteration 8 Loss: -28.4519
            Iteration 9 Loss: -25.9762
    ```
8. Then, all the updated weights of model can be seen, which are very different from the init weights.
    ```
    MLP Architecutre & weight after training

    Layer0: 
    Layer Weights: 18   
    weights: 0.852243, -0.159087, bias: 0.705576
    weights: 0.549085, -0.143006, bias: -0.0916505
    weights: 0.278763, 0.201569, bias: -0.292848
    weights: 0.260602, 0.891634, bias: 0.531036
    weights: 1.0325, 0.422959, bias: 0.146165
    weights: -1.04009, -0.518657, bias: -0.111508

    Layer1: 
    Layer Weights: 21
    weights: 0.368325, -0.819267, 0.87151, 0.238725, -0.803273, -0.631082, bias: 0.264608
    weights: 0.794461, -0.0174709, -0.894927, -0.172642, 0.631393, 0.460614, bias: 0.0574533
    weights: -1.14616, -0.251609, 0.731876, -1.07848, -0.798524, 0.242482, bias: -0.606092

    Layer2: 
    Layer Weights: 8
    weights: -0.563835, -0.418773, 0.641282, bias: -0.5
    weights: -0.100595, 0.195498, 1.50782, bias: -0.5

    Total MLP Weights: 47
    ```
9. Then, a test is done on 10 random samples.
10. finally a test accuracy is printed. 
    ```
    Test Accuracy: 60%
    ```