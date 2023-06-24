#include <functional>
#include <unordered_set>
#include <string>
#include <iostream>
#include <memory>
#include <vector>
#include <cmath>
#include "engine.h"

/**
    * @brief Value class implements the fundamental building block of an autograd engine.

    * For simplicity, it accepts only scalars.
    * You can create a Value object simply by wrapping around any float.
    * For ex. wrap a scalar, 2.5 in a Value class via `auto v1 = std::make_shared<Value>(2.5);`
    * Once you create a Value object for a scalar, it can be included as a node in the bigger neural network graph.
    * For more intuitive understanding of a node, its role in a nerual network graph, and how back prop comes into the picture check out `digin-micrograd-theory`.

    * @param data (type: float): The scalar value wrapped in the Value object.
    * @param grad (type: float): gradient of the final node in the autograd graph, wrt the current value object.
    * @param prev (type: std::unordered_set<std::shared_ptr<Value>>): The set of Value objects that created the current Value object.
    * @param op (type: std::string): The operation (like +, *) that was performed to create current value object.
    * @param _backward: A lambda function representing the expression to calculate the derivative of the final node with respect to the current node (this Value object).

*/

Value::Value(float data, std::unordered_set<std::shared_ptr<Value>> prev, std::string op) {
    this->data = data;
    this->grad = 0.0;
    this->prev = std::move(prev);
    this->op = std::move(op);
    this->_backward = [this] {
        for (const auto& child : this->prev) {
            child->_backward();
        }
    };
}

/**
     * @brief Retrieves the scalar value stored in the Value object.
     * @return The scalar value (type: float) wrapped in the Value object.
*/
float Value::get_data() {
    return data;
}

/**
     * @brief Sets the scalar value stored in the Value object.
*/
void Value::set_data(float data) {
    this->data = data;
}


/**
     * @brief Retrieves the set of Value objects that created the current Value object.
     * @return The set of Value objects (type: std::unordered_set<std::shared_ptr<Value>>) that created the current Value object.
*/
std::unordered_set<std::shared_ptr<Value>> Value::get_prev() const {
    return prev;
}

/**
     * @brief Retrieves the gradient value associated with the Value object.
     * @return The gradient value (type: float) associated with the Value object.
*/
float Value::get_grad() const {
    return grad;
}

/**
     * @brief Sets the gradient value for the Value object.
     * @param grad_value The gradient value (type: float) to be set.
*/
void Value::set_grad(float grad_value) {
    this->grad=grad_value;
}

/**
     * @brief Overloaded operator for addition of two Value objects.
     * For ex. 
     * auto v1 = std::make_shared<Value>(2.5);
     * auto v2 = std::make_shared<Value>(3.5);
     * auto v1_2 = v1+v2;
     * defining the operator+ allows us to use the intuitive expression a+b.

     * @param other The other Value object to be added.
     * @return A new Value object (type: std::shared_ptr<Value>) representing the sum of the two Value objects.
*/
std::shared_ptr<Value> Value::operator+(const std::shared_ptr<Value>& other) {
    auto out_prev = std::unordered_set<std::shared_ptr<Value>>{shared_from_this(), other};

    auto out = std::make_shared<Value>(data + other->data, out_prev, "+");

    out->_backward = [this, other, out] {
        grad += out->grad;
        other->grad += out->grad;

    };
    return out;
}

/**
     * @brief Overloaded operator for negation of the Value object.
     * For ex. 
     * auto v1 = std::make_shared<Value>(2.5);
     * auto v1 = -v1;
     * defining the operator- allows us to use the intuitive expression (-b).

     * @return A new Value object (type: std::shared_ptr<Value>) representing the negated Value object.
*/
std::shared_ptr<Value> Value::operator-() {
    return shared_from_this() * std::make_shared<Value>(-1.0);
}

/**
     * @brief Overloaded operator for subtraction of two Value objects.
     * For ex. 
     * auto v1 = std::make_shared<Value>(2.5);
     * auto v2 = std::make_shared<Value>(3.5);
     * auto v1_2 = v1-v2;
     * defining the operator- allows us to use the intuitive expression a+(-b).

     * @param other The other Value object to be subtracted.
     * @return A new Value object (type: std::shared_ptr<Value>) representing the subtraction of the two Value objects.
*/
std::shared_ptr<Value> Value::operator-(const std::shared_ptr<Value>& other) {
    return shared_from_this() + (other->operator-());
}

/**
     * @brief Overloaded operator for power of two Value objects.
     * For ex. 
     * auto v1 = std::make_shared<Value>(2.5);
     * auto v2 = std::make_shared<Value>(3.5);
     * auto v1_2 = v1->pow(v2);
     * defining the pow allows us to use the intuitive expression a->pow(b).

     * @param other The other Value object which acts as the power.
     * @return A new Value object (type: std::shared_ptr<Value>) representing v1^v2.
*/
std::shared_ptr<Value> Value::pow(const std::shared_ptr<Value>& other) {
    auto out_prev = std::unordered_set<std::shared_ptr<Value>>{shared_from_this(), other};

    auto out = std::make_shared<Value>(std::pow(data, other->data), out_prev, "^");

    out->_backward = [this, other, out] {
        grad +=other->data * std::pow(data, other->data - 1) * out->grad;

    };
    return out;
}

/**
     * @brief Overloaded operator for division of two Value objects.
     * For ex. 
     * auto v1 = std::make_shared<Value>(2.5);
     * auto v2 = std::make_shared<Value>(3.5);
     * auto v1_2 = v1/v2;
     * defining the operator/ allows us to use the intuitive expression a/b.

     * @param other The other Value object to be divided.
     * @return A new Value object (type: std::shared_ptr<Value>) representing the division of the two Value objects.
*/
std::shared_ptr<Value> Value::operator/(const std::shared_ptr<Value>& other) {
    return shared_from_this() * other->pow(std::make_shared<Value>(-1)) ;
}

/**
     * @brief Overloaded operator for multiplication of two Value objects.
     * For ex. 
     * auto v1 = std::make_shared<Value>(2.5);
     * auto v2 = std::make_shared<Value>(3.5);
     * auto v1_2 = v1*v2;
     * defining the operator* allows us to use the intuitive expression a*b.

     * @param other The other Value object to be multiplied.
     * @return A new Value object (type: std::shared_ptr<Value>) representing the product of the two Value objects.
*/
std::shared_ptr<Value> Value::operator*(const std::shared_ptr<Value>& other) {
    auto out_prev = std::unordered_set<std::shared_ptr<Value>>{shared_from_this(), other};

    auto out = std::make_shared<Value>(data * other->data, out_prev, "*");

    out->_backward = [this, other, out] {
        grad += other->data * out->grad;
        other->grad += data * out->grad;
        
    };
    return out;
}

/**
     * @brief Performs the backward pass for automatic differentiation using backpropagation.
     * Calculates the gradients for all the Value objects in the computation graph.
     * Gradient of the top-most node is calculated first, and then correspondingly for lower nodes, via chain-rule implemented in each node's _backward function.
     * For deeper intuition checkout `digin-micrograd-theory`.
     
*/
void Value::backward() {
    std::vector<std::shared_ptr<Value>> topo;
    std::unordered_set<std::shared_ptr<Value>> visited;

    std::function<void(const std::shared_ptr<Value>&)> build_topo = [&](const std::shared_ptr<Value>& v) {
        if (visited.find(v) == visited.end()) {
            visited.insert(v);

            for (const auto& child : v->prev) {
                build_topo(child);
            }
            topo.push_back(v);
        }
    };

    build_topo(shared_from_this());

    grad = 1.0;

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        const auto& v = *it;
        v->_backward();
    }
}

// Non-member operators for global-level access to expressing a+b etc..

/**
 * @brief Overloaded operator for addition of two Value objects.
 * @param lhs The left-hand side Value object.
 * @param rhs The right-hand side Value object.
 * @return A new Value object (type: std::shared_ptr<Value>) representing the sum of the two Value objects.
 */
std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
    return (*lhs) + rhs;
}

/**
 * @brief Overloaded operator for subtraction of two Value objects.
 * @param lhs The left-hand side Value object.
 * @param rhs The right-hand side Value object.
 * @return A new Value object (type: std::shared_ptr<Value>) representing the subtraction of the two Value objects.
 */
std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
    return (*lhs) - rhs;
}

/**
 * @brief Overloaded operator for multiplication of two Value objects.
 * @param lhs The left-hand side Value object.
 * @param rhs The right-hand side Value object.
 * @return A new Value object (type: std::shared_ptr<Value>) representing the product of the two Value objects.
 */
std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
    return (*lhs) * rhs;
}

/**
 * @brief Overloaded operator for division of two Value objects.
 * @param lhs The left-hand side Value object.
 * @param rhs The right-hand side Value object.
 * @return A new Value object (type: std::shared_ptr<Value>) representing the division of the two Value objects.
 */
std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
    return (*lhs) / rhs;
}

/**
 * @brief Overloaded operator for power of two Value objects.
 * @param lhs The left-hand side Value object.
 * @param rhs The right-hand side Value object.
 * @return A new Value object (type: std::shared_ptr<Value>) representing the power of the two Value objects.
 */
std::shared_ptr<Value> pow(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
    return lhs->pow(rhs);
}