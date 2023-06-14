// Vvery very basic C++ implementation. Hence, scratch!
// Right now, only supports ADD operator.
// This script will slowly incorporate more operations.
// This script is far from optimized, 
// so will need dedicated look for optimizations, 
// once it works e2e for all operations first.

#include <functional>
#include <unordered_set>
#include <string>
#include <iostream>
#include <memory>


class Value : public std::enable_shared_from_this<Value> {
private:
    // Private member variables
    double data;
    double grad;
    std::function<void()> backward;
    std::unordered_set<std::shared_ptr<Value>> prev;
    std::string op;

public:
    // Constructor
    Value(double data, std::unordered_set<std::shared_ptr<Value>> prev = {}, std::string op = ""){
        this->data = data;
        this->grad = 0.0;
        this->prev = std::move(prev);
        this->op = std::move(op);
    }
    // Public getter for private data member variable
    double get_data() const {
        return data;
    }
    // Overriden logic of ADD and MUL operators
    std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& other) {
        prev = std::unordered_set<std::shared_ptr<Value>>{shared_from_this(), other};

        auto out = std::make_shared<Value>(data+other->data, prev, "+");

        out->backward = [this, other, out] {
            grad += out->grad;
            other->grad += out->grad;
        };
        return out;
    }
    std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& other){
        prev = std::unordered_set<std::shared_ptr<Value>>{shared_from_this(), other};

        auto out = std::make_shared<Value>(data*other->data, prev, "*");

        out->backward = [this, other, out] {
            grad += other->data*out->data;
            other->grad += data*out->data;
        };
        return out;
    }
};

//  Defining skeleton outside Value Class scope to support the syntax value1 + value2 and value1 * value2,
//  where value1 and value2 are objects of the Value class.
//  sort-of what happens: *lhs dereferences to std::shared_ptr<Value> and then calls Value's operator+ with rhs as argument.

std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
    return (*lhs) + rhs;
}
std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
    return (*lhs) * rhs;
}


int main() {
    // Create two Value objects
    // Value value1(2.5);
    // Value value2(3.7);
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