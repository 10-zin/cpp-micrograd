#include "engine.h"
#include "nn.h"
#include <vector>

int main(){
    int nin=3;
    std::vector<int> nout {6, 3, 1};
    
    auto mlp = MLP(nin, nout);
    std::vector<std::shared_ptr<Value>> inputs;
    for (int i=0; i < 3; ++i){
        std::shared_ptr<Value> inp_val = std::make_shared<Value>(i);
        inputs.push_back(inp_val);
    } 
    auto output = mlp(inputs);
    for(auto op: output){
        std::cout<<"MLP O/P: ";
        std::cout<<op->get_data();
    }
    mlp.show_parameters();
    std::cout<<"\nTotal MLP Weights: "<<mlp.parameters().size()<<std::endl;
    output[0]->backward();
    return 0;
};