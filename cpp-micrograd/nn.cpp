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

int main(){
    Neuron neuron(3, true);
    neuron.show_parameters();
    neuron.zero_grad();
    
    return 0;
};


// class Layer {
//     private:
//         std::vector<Neuron> neurons;
// };

// class MLP {
//     private:
//         int nin;
//         std::vector<int> nouts;
//         std::vector<Layer> layers;
