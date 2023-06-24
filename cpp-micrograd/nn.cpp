#include "engine.h"
#include "nn.h"
#include <iostream>
#include<vector>
#include <random>

/**
 * @brief Module class
 *
 * The Module class represents a base class for neural network modules.
 * It provides methods for zeroing out the gradients of the parameters.
 */

void Module::zero_grad(){
    for (auto& weight: parameters()){
        weight->set_grad(0.0);
    }
}

/**
 * @brief Neuron class
 *
 * The Neuron class represents a single neuron in a neural network layer.
 * It holds the weights and bias associated with the neuron and provides
 * functionality for computing the output of the neuron.
 */
Neuron::Neuron (int nin, bool nonlin){
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

std::shared_ptr<Value> Neuron::operator()(std::vector<std::shared_ptr<Value>>& x){
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

void Neuron::show_parameters() {
    std::cout << "weights: ";
    for (auto& weight: this->weights){
        std::cout << weight->get_data() << ", ";
    }
    std::cout<<"bias: "<<bias->get_data()<<std::endl;
}

std::vector<std::shared_ptr<Value>> Neuron::parameters() {
    std::vector<std::shared_ptr<Value>> parameters;
    parameters.reserve(weights.size() + 1);

    for (auto& weight : weights) {
        parameters.emplace_back(weight);
    }
    parameters.emplace_back(bias);

    return parameters;
}

/**
 * @brief Layer class
 * 
 * The Layer class represents a layer in a neural network.
 * It consists of multiple neurons and provides functionality for computing
 * the output of the layer and accessing the layer's parameters.
*/
Layer::Layer(int nin, int nout){
    total_params=(nin+1)*nout;
    neurons.reserve(nout+1);

    for (int i=0; i< nout; ++i){
        Neuron neuron(nin, true);
        neurons.emplace_back(neuron);
    }
}

std::vector<std::shared_ptr<Value>> Layer::operator()(std::vector<std::shared_ptr<Value>> x){
    std::vector<std::shared_ptr<Value>> out;
    out.reserve(neurons.size()+1);
    for (auto& neuron: neurons){
        out.emplace_back(neuron(x));
    }
    return out;
}

std::vector<std::shared_ptr<Value>> Layer::parameters() {
    std::vector<std::shared_ptr<Value>> parameters;
    parameters.reserve(total_params + 1);

    for (auto neuron : neurons) {
        for(auto weight:  neuron.parameters()){
            parameters.emplace_back(weight);
        }
    }

    return parameters;
}

void Layer::show_parameters() {
    int i=0;
    std::cout<<"Layer Weights: "<<total_params<<std::endl;
    for (auto neuron : neurons) {
        neuron.show_parameters();
    }
}


/**
 * @brief MLP class
 * 
 * The MLP class represents a Multi-Layer Perceptron neural network.
 * It consists of multiple layers and provides functionality for
 * computing the output of the network and accessing the network's parameters.
*/

MLP::MLP(int nin, std::vector<int> nout) {
    layers.reserve(nout.size()+1);
    total_params=0;

    for (int i=0; i<nout.size(); ++i){
        if (i==0){
            layers.emplace_back(Layer(nin, nout[i]));
            total_params=total_params+nin*nout[i];
        }
        else{
            layers.emplace_back(Layer(nout[i-1], nout[i]));
            total_params=total_params+nout[i-1]*nout[i];
        }
    }

}

std::vector<std::shared_ptr<Value>> MLP::operator()(std::vector<std::shared_ptr<Value>> x){
    for (auto layer: layers){
        x = layer(x);
    }
    return x;
}

std::vector<std::shared_ptr<Value>> MLP::parameters() {
    std::vector<std::shared_ptr<Value>> parameters;
    parameters.reserve(total_params + 1);

    for (auto layer : layers) {
        for(auto param:  layer.parameters()){
            parameters.emplace_back(param);
            
        }
    }

    return parameters;
}


void MLP::show_parameters() {
    int i =0;
    for (auto layer : layers) {
        std::cout<<"\nLayer"<<i<<": "<<std::endl;
        layer.show_parameters();
        i=i+1;
    }
}
