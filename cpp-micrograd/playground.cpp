#include "engine.h"
#include <iostream>

int main() {
    // Use the Value class here
    // ...
    std::shared_ptr<Value> value1 = std::make_shared<Value>(2.5);
    std::shared_ptr<Value> value2 = std::make_shared<Value>(3.7);
    std::shared_ptr<Value> value3 = std::make_shared<Value>(-3.0);
    std::shared_ptr<Value> value4 = std::make_shared<Value>(1.7);

    // Perform addition using the operator+
    std::shared_ptr<Value> value1_2 = value1 + value2;

    // Perform multiplication using the operator*
    std::shared_ptr<Value> value_1_2_3 = value1_2 * value3;

    std::shared_ptr<Value> result_final = value_1_2_3+value4;


    // Access the result and print the data
    std::cout << "Result Final: " << result_final->get_data() << std::endl;
    // result_final->set_grad(1.0);
    result_final->backward();
    std::cout<<"\n\n\n";
    std::cout<<value1->get_grad();
    std::cout<<value2->get_grad();
    std::cout<<value3->get_grad();
    std::cout<<value4->get_grad();
    // std::cout << "Result MUL: " << result_mul->get_data() << std::endl;
    
    return 0;
}