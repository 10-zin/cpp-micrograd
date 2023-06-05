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
public:
    double data;
    double grad;
    std::function<void()> backward;
    std::unordered_set<std::shared_ptr<Value>> prev;
    std::string op;

    static std::shared_ptr<Value> create(double data, std::unordered_set<std::shared_ptr<Value>> prev = {}, std::string op = "") {
        return std::make_shared<Value>(Value(data, std::move(prev), std::move(op)));
    }

private:
    Value(double data, std::unordered_set<std::shared_ptr<Value>> prev, std::string op)
        : data(data), grad(0.0), prev(std::move(prev)), op(std::move(op)) {}

public:
    std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& other) {
        auto out = Value::create(data + other->data, std::unordered_set<std::shared_ptr<Value>>{shared_from_this(), other}, "+");
        out->backward = [this, other, out] {
            grad += out->grad;
            other->grad += out->grad;
        };
        return out;
    }
    std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& other){
        auto out = Value::create(data*other->data, std::unordered_set<std::shared_ptr<Value>>{shared_from_this(), other}, "*");
        out->backward = [this, other, out] {
            grad += other->data*out->data;
            other->grad += data*out->data;
        };
        return out;
    }

};

std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
    return (*lhs) + rhs;
}
std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
    return (*lhs) * rhs;
}


int main() {
    // Create two Value objects
    std::shared_ptr<Value> value1 = Value::create(2.5);
    std::shared_ptr<Value> value2 = Value::create(3.7);

    // Perform addition using the operator+
    std::shared_ptr<Value> result_add = value1 + value2;

    // Perform multiplication using the operator*
    std::shared_ptr<Value> result_mul = value1 * value2;

    // Access the result and print the data
    std::cout << "Result ADD: " << result_add->data << std::endl;
    std::cout << "Result MUL: " << result_mul->data << std::endl;

    return 0;
}