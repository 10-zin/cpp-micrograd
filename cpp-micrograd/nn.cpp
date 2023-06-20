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
        std::shared_ptr<Value> operator()(std::vector<std::shared_ptr<Value>>& x){
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

class Layer: public Module{
    private:
        std::vector<Neuron> neurons;
        int total_params;

    public:
        Layer(int nin, int nout){
            total_params=nin*nout;
            neurons.reserve(neurons.size()+1);

            for (int i=0; i< nout; ++i){
                Neuron neuron(nin, true);
                neurons.emplace_back(neuron);
            }
        }

        std::vector<std::shared_ptr<Value>> operator()(std::vector<std::shared_ptr<Value>> x){
            std::vector<std::shared_ptr<Value>> out;
            out.reserve(neurons.size()+1);
            for (auto& neuron: neurons){
                out.emplace_back(neuron(x));
            }
            return out;
        }

        std::vector<std::shared_ptr<Value>> parameters() override {
            std::vector<std::shared_ptr<Value>> parameters;
            parameters.reserve(total_params + 1);

            for (auto neuron : neurons) {
                for(auto param:  neuron.parameters()){
                    parameters.emplace_back(param);
                }
            }

            return parameters;
        }

};

class MLP: public Module{
    private:
        std::vector<Layer> layers;
        int total_params;
    public:
        MLP(int nin, std::vector<int> nout) {
            layers.reserve(nout.size()+1);
            total_params=1;

            for (int i=0; i<nout.size(); ++i){
                if (i==0){
                    layers.emplace_back(Layer(nin, nout[i]));
                    total_params=total_params*nin*nout[i];
                }
                else{
                    layers.emplace_back(Layer(nout[i-1], nout[i]));
                    total_params=total_params*nout[i-1]*nout[i];
                }
            }
        }

        std::vector<std::shared_ptr<Value>> operator()(std::vector<std::shared_ptr<Value>> x){
            for (auto layer: layers){
                x = layer(x);
            }
            return x;
        }

        std::vector<std::shared_ptr<Value>> parameters() override {
            std::vector<std::shared_ptr<Value>> parameters;
            parameters.reserve(total_params + 1);

            for (auto layer : layers) {
                for(auto param:  layer.parameters()){
                    parameters.emplace_back(param);
                }
            }

            return parameters;
        }

};

int main(){
    int nin=3;
    std::vector<int> nout {6, 3, 1};
    // Neuron neuron(3, true);
    // neuron.show_parameters();
    auto mlp = MLP(nin, nout);
    std::vector<std::shared_ptr<Value>> inputs;
    for (int i=0; i < 3; ++i){
        std::shared_ptr<Value> inp_val = std::make_shared<Value>(i);
        inputs.push_back(inp_val);
    } 
    auto output = mlp(inputs);
    for(auto op: output){
        std::cout<<"MLP O/P";
        std::cout<<op->get_data();
    }
    return 0;
};


