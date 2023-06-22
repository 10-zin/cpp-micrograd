#include <functional>
#include <unordered_set>
#include <string>
#include <iostream>
#include <memory>
#include <vector>
#include "engine.h"

// Implementation of the Value class member functions

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