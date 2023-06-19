#include "engine.h"
#include <iostream>
#include<vector>
#include <random>

class Module {
    public:
        void zero_grad(){
            for (const auto& weight: parameters()){
                weight->zero_grad();
            }
        }

        virtual std::vector<std::shared_ptr<Value>> parameters() = 0;

};

class Neuron: public Module{
    private:
        std::vector<std::shared_ptr<Value>> weights;
        std::shared_ptr<Value> bias = std::make_shared<Value>(0);
        bool nonlin;

    public:
        Neuron (int nin, bool nonlin=true){
            this->nonlin = nonlin;

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_real_distribution<> dis(-1.0, 1.0);
            this->weights.reserve(nin);
            for (int i = 0; i < nin; ++i) {
                auto weight = std::make_shared<Value>(dis(gen));
                this->weights.emplace_back(weight);
            }
        }
        std::shared_ptr<Value> operator()(const std::vector<std::shared_ptr<Value>>& x){
            std::shared_ptr<Value> act = std::make_shared<Value>(0.0);;
            for (int i=0; i<x.size(); ++i){
                act = act + (x[i]*weights[i]);
            }
            act = act+bias;
            
            if (nonlin) {
                // return act.relu();
                return act;
            }
            return act;

        }
        void show_parameters() {
            for (auto& weight: this->weights){
                std::cout << weight->get_data() << std::endl;
            }
        }
        std::vector<std::shared_ptr<Value>> parameters() override {
            std::vector<std::shared_ptr<Value>> parameters;
            parameters.reserve(weights.size() + 1);

            for (const auto& weight : weights) {
                parameters.emplace_back(weight);
            }
            parameters.emplace_back(bias);

            return parameters;
    }
    
};

// class Layer {
//     private:
//         std::vector<Neuron> neurons;

//     public:
//         Layer(int nout, int nin){
//             neurons.reserve(neurons.size()+1);

//             for (int i=0; i< nout; ++i){
//                 Neuron neuron(nin, true);
//                 neurons.emplace_back(neuron);
//             }
//         }

// };

int main(){
    Neuron neuron(3, true);
    neuron.show_parameters();
    std::vector<std::shared_ptr<Value>> inputs;
    for (int i=0; i < 3; ++i){
        std::shared_ptr<Value> inp_val = std::make_shared<Value>(i);
        inputs.push_back(inp_val);
    } 
    auto act = neuron(inputs);
    std::cout<<act->get_data();
    neuron.zero_grad();
    return 0;
};

// class MLP {
//     private:
//         int nin;
//         std::vector<int> nouts;
//         std::vector<Layer> layers;
