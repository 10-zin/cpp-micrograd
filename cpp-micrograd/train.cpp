#include "engine.h"
#include "nn.h"
#include <vector>
#include <cmath>
#include <cstdlib>

int main(){
    /** 
     * @brief A neural network that models the binary AND operation.
     * Let's train a neural network to model the binary AND operation.
     * Expected behaviour:
        * Input | output
        * 0 0   | 0
        * 0 1   | 0
        * 1 0   | 0
        * 1 1   | 1
    */

    /**
     * @brief Create a MLP architecture for modelling the AND operation.
     * a simple fully connected neural network with multiple layers, is known as a multi layer perceptron (MLP).
     * Important:
     * Since AND operation takes two values, the first layer must have only 2 neurons.
     * the `nin` variable stores number of neurons in the input layer.
     * Then we can specify number of neurons in the remaining layer as we like.
     * In this case: l1: 2 -> l2: 6 -> l3: 3 
     * Now the last layer can have either one or two neurons. One neuron makes it a regression problem, 2 makes it a classification problem.
     * Let's take 2 neurons.
     * Final architecture:
     * l1: 2 -> l2: 6 -> l3: 3 -> l4: 2
    */
    int nin=2;
    std::vector<int> nout {6, 3, 2};
    
    // Now, initialize a maulti-layer perceptron, a very simple neural network.
    auto mlp = MLP(nin, nout);
    std::cout<<"MLP Architecutre & weight upon initialization"<<std::endl;
    mlp.show_parameters();
    std::cout<<"\nTotal MLP Weights: "<<mlp.parameters().size()<<std::endl;

    /**
     * @brief Training Dataset Creation
     * Let's create a training dataset, for training the MLP.
     * Each training example, will be a set of input, target.
     * input and target both will be a 2 dim vector.
    */
    int num_train = 10;
    std::vector<std::tuple<std::vector<std::shared_ptr<Value>>, std::vector<std::shared_ptr<Value>>>> train_set;
    for (int i=0; i < num_train; ++i){
        std::vector<std::shared_ptr<Value>> operands;
        std::vector<std::shared_ptr<Value>> label;
        float op1 = rand()%2;
        float op2 = rand()%2;
        auto operand_1 = std::make_shared<Value>(op1);
        auto operand_2 = std::make_shared<Value>(op2);
        operands.push_back(operand_1);
        operands.push_back(operand_2);

        if (op1 && op2){
            auto prob_0 = std::make_shared<Value>(0.0);
            auto prob_1 = std::make_shared<Value>(1.0);
            label.push_back(prob_0);
            label.push_back(prob_1);
        }
        else{
            auto prob_0 = std::make_shared<Value>(1.0);
            auto prob_1 = std::make_shared<Value>(0.0);
            label.push_back(prob_0);
            label.push_back(prob_1);
        }
        std::tuple<std::vector<std::shared_ptr<Value>>, std::vector<std::shared_ptr<Value>>> train_example(operands, label);
        train_set.push_back(train_example);
    } 

    /**
     * @brief Training Loop
     * do one loop for the entire training set.
     * feed each input to mlp, get the output from mlp which will be a 2 dim vector.
     * the target is also a 2 dim vector.
     * calculate the loss, (prediction[i]-target[i])^2 where i is 0, and 1.
     * Mean squared error is a simple loss function we can take.
     * then do loss->backward(). As loss is the final value object created in the entire computation graph.
     * This will calulate gradient for all weights in the mlp, hence an `autograd engine`.
     * Then update all weights by doing w_new = w-lr*grad.
     * The gradient will guide the weights such that the overall loss reduces.
     * And as we see the loss gradually decreases!
    */
    std::cout<<"\nTraining loop:"<<std::endl;
    std::shared_ptr<Value> final_loss;
    float learning_rate = 0.1;
    int i=0;
    for (auto& train_example: train_set){
        auto operands = std::get<0>(train_example);
        auto target = std::get<1>(train_example);

        auto prediction = mlp(operands);
        std::shared_ptr<Value> total_loss = std::make_shared<Value>(0.0);
        for (int i=0; i<target.size(); ++i){
            auto loss = prediction[i]-target[i];
            loss->pow(std::make_shared<Value>(2));
            total_loss = total_loss+loss;
        }
        final_loss = total_loss / std::make_shared<Value>(target.size());

        mlp.zero_grad();
        final_loss->backward();
        
        for (auto param : mlp.parameters()) {
            auto updated_weight = param->get_data()-learning_rate * param->get_grad();
            param->set_data(updated_weight);
        }
        std::cout<<"Iteration "<<i<<" Loss: "<<final_loss->get_data()<<std::endl;
        i+=1;
    }

    /**
     * @brief Checkout updated weights
     * If the weights updated correctly all the weights will be different from when they were initialized.
    */
    std::cout<<"MLP Architecutre & weight after training"<<std::endl;
    mlp.show_parameters();
    std::cout<<"\nTotal MLP Weights: "<<mlp.parameters().size()<<std::endl;

    /**
     * @brief Test loop
     * Same as the training loop.
     * Just that we want to note which of the dimension in prediction vector scores the largest.
     * the dimension that scores the largest is the model's output of the input operands.
     * The more accurate the prediction, the more the model learns the AND gate logic.
    */
    std::cout<<"Now testing...\n\n";
    int num_test = 10;
    std::vector<std::tuple<std::vector<std::shared_ptr<Value>>, std::vector<std::shared_ptr<Value>>>> test_set;
    int num_correct_preds=0;
    for (int i=0; i < num_test; ++i){
        std::vector<std::shared_ptr<Value>> operands;
        float label;
        float op1 = rand()%2;
        float op2 = rand()%2;
        auto operand_1 = std::make_shared<Value>(op1);
        auto operand_2 = std::make_shared<Value>(op2);
        operands.push_back(operand_1);
        operands.push_back(operand_2);

        if (op1 && op2){
            label = 1;
        }
        else{
            label = 0;
        }
        auto prediction = mlp(operands);
        float predicted_value;
        // std::cout<<prediction[0]->get_data()<<prediction[1]->get_data()<<std::endl;
        if (prediction[0]>prediction[1]){
            predicted_value=0;
        }
        else{
            predicted_value=1;
        }
        if (predicted_value==label){
            num_correct_preds+=1;
        }
        // std::cout<<predicted_value<<label<<std::endl;
    }
    float accuracy;
    accuracy = (static_cast<float>(num_correct_preds)/num_test)*100;
    std::cout<<"Test Accuracy: "<<accuracy<<"%"<<std::endl;
    return 0;
};