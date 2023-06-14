#include "engine.h"
#include <iostream>

int main() {
    // Use the Value class here
    // ...
    std::shared_ptr<Value> value1 = std::make_shared<Value>(2.5);
    std::shared_ptr<Value> value2 = std::make_shared<Value>(3.7);

    // Perform addition using the operator+
    std::shared_ptr<Value> result_add = value1 + value2;

    // Perform multiplication using the operator*
    std::shared_ptr<Value> result_mul = value1 * value2;

    // Access the result and print the data
    std::cout << "Result ADD: " << result_add->get_data() << std::endl;
    std::cout << "Result MUL: " << result_mul->get_data() << std::endl;
    
    return 0;
}