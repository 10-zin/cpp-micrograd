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

    /**
     * @brief Training Dataset Creation
     * Let's create a training dataset, for training the MLP.
     * Each training example, will be a set of input, target.
     * input will be a 2 dim vector, and target will be 1 dim.
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

    // feed input vector to the mlp.
    // the mlp predicts one value (as the last layer has one neuron)
    std::shared_ptr<Value> final_loss;
    float learning_rate = 0.1;
    int i=0;
    for (auto& train_example: train_set){
        auto operands = std::get<0>(train_example);
        auto target = std::get<1>(train_example);

        auto result = mlp(operands);
        std::shared_ptr<Value> total_loss = std::make_shared<Value>(0.0);
        for (int i=0; i<target.size(); ++i){
            auto loss = result[i]-target[i];
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


    mlp.show_parameters();
    std::cout<<"\nTotal MLP Weights: "<<mlp.parameters().size()<<std::endl;

    // test
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
        std::cout<<prediction[0]->get_data()<<prediction[1]->get_data()<<std::endl;
        if (prediction[0]>prediction[1]){
            predicted_value=0;
        }
        else{
            predicted_value=1;
        }
        if (predicted_value==label){
            num_correct_preds+=1;
        }
        std::cout<<predicted_value<<label<<std::endl;
    }
    float accuracy;
    accuracy = (static_cast<float>(num_correct_preds)/num_test)*100;
    std::cout<<"Accuracy: "<<accuracy<<"%"<<std::endl;
    return 0;
};