#include <functional>
#include <unordered_set>
#include <string>
#include <iostream>
#include <memory>
#include <vector>
#include "engine.h"

/*
* @brief Value class implements the fundamental building block of an autograd engine.
* 
* For simplicity, it accepts only scalars.
* You can create a Value object simply by wrapping around any float.
* For ex. wrap a scalar, 2.5 in a Value class via `auto v1 = std::make_shared<Value>(2.5);`
* Once you create a Value object for a scalar, it can be included as a node in the bigger neural network graph.
* For more intuitive understanding of a node, its role in a nerual network graph, and how back prop comes into the picture check out `digin-micrograd-theory`.
*
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

float Value::get_data() const {
    return data;
}

std::unordered_set<std::shared_ptr<Value>> Value::get_prev() const {
    return prev;
}

float Value::get_grad() const {
    return grad;
}

void Value::set_grad(float grad_value) {
    this->grad=grad_value;
}

std::shared_ptr<Value> Value::operator+(const std::shared_ptr<Value>& other) {
    auto out_prev = std::unordered_set<std::shared_ptr<Value>>{shared_from_this(), other};

    auto out = std::make_shared<Value>(data + other->data, out_prev, "+");

    out->_backward = [this, other, out] {
        std::cout<<"\n\n----BACKWARD(+)---\n";
        std::cout<<"\nout "<<out->get_data()<<" this "<<this->get_data()<<" other "<<other->get_data();
        std::cout<<"\nBefore:\n"<<"grad: "<<grad<<" out->grad: "<<out->grad<<"other->grad: "<<other->grad<<std::endl;
        grad += out->grad;
        other->grad += out->grad;
        std::cout<<"\nAfter:\n"<<"grad: "<<grad<<" out->grad: "<<out->grad<<"other->grad: "<<other->grad<<std::endl;
        std::cout<<"\n\n----BACKWARD(+) END---\n";
    };
    return out;
}

std::shared_ptr<Value> Value::operator*(const std::shared_ptr<Value>& other) {
    auto out_prev = std::unordered_set<std::shared_ptr<Value>>{shared_from_this(), other};

    auto out = std::make_shared<Value>(data * other->data, out_prev, "*");

    out->_backward = [this, other, out] {
        std::cout<<"\n\n---BACKWARD(*)---\n";
        std::cout<<"\nBefore:\n"<<"grad: "<<grad<<" other->data: "<<other->data<<"out->data: "<<out->data<<std::endl;
        std::cout<<"\nBefore:\n"<<"other->grad: "<<other->grad<<"data: "<<data<<"out->data: "<<out->data<<std::endl;
        grad += other->data * out->grad;
        other->grad += data * out->grad;
        std::cout<<"\nAfter:\n"<<"grad: "<<grad<<" other->data: "<<other->data<<"out->data: "<<out->data<<std::endl;
        std::cout<<"\nAfter:\n"<<"other->grad: "<<other->grad<<"data: "<<data<<"out->data: "<<out->data<<std::endl;
        std::cout<<"\n---BACKWARD(*) END---\n";
        
    };
    return out;
}

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

    grad = 1.0f;
    std::cout<<"Topo elements backprop order(will start from right end)"<<std::endl;
    for (auto v: topo){
        std::cout<<v->data<<" ";
    }

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        const auto& v = *it;
        std::cout<<"\n\nnode\n";
        std::cout<<"data: "<<v->data<<" grad: "<<v->data;
        v->_backward();
    }
}


// Implementation of the non-member operators

std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
    return (*lhs) + rhs;
}

std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
    return (*lhs) * rhs;
}