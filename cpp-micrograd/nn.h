#ifndef NN_H
#define NN_H

#include "engine.h"
#include <iostream>
#include<vector>
#include <random>

class Module {
    public:
        void zero_grad();
        virtual std::vector<std::shared_ptr<Value>> parameters()=0;

};

class Neuron: public Module{
    private:
        std::vector<std::shared_ptr<Value>> weights;
        std::shared_ptr<Value> bias = std::make_shared<Value>(0);
        bool nonlin;

    public:
        Neuron (int nin, bool nonlin=true);
        std::shared_ptr<Value> operator()(std::vector<std::shared_ptr<Value>>& x);
        std::vector<std::shared_ptr<Value>> parameters() override;
        void show_parameters();
    
};

class Layer: public Module{
    private:
        std::vector<Neuron> neurons;
        int total_params;

    public:
        Layer(int nin, int nout);
        std::vector<std::shared_ptr<Value>> operator()(std::vector<std::shared_ptr<Value>> x);
        std::vector<std::shared_ptr<Value>> parameters() override ;
        void show_parameters() ;

};

class MLP: public Module{
    private:
        std::vector<Layer> layers;
        int total_params;
    public:
        MLP(int nin, std::vector<int> nout) ;
        std::vector<std::shared_ptr<Value>> operator()(std::vector<std::shared_ptr<Value>> x);
        std::vector<std::shared_ptr<Value>> parameters() override ;
        void show_parameters() ;

};

#endif
