#include <functional>
#include <unordered_set>
#include <string>
#include <iostream>
#include <memory>
#include <vector>
#include "engine.h"

// Implementation of the Value class member functions

Value::Value(double data, std::unordered_set<std::shared_ptr<Value>> prev, std::string op) {
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

double Value::get_data() const {
    return data;
}

void Value::zero_grad() {
    this->grad=0.0;
}

std::shared_ptr<Value> Value::operator+(const std::shared_ptr<Value>& other) {
    prev = std::unordered_set<std::shared_ptr<Value>>{shared_from_this(), other};

    auto out = std::make_shared<Value>(data + other->data, prev, "+");

    out->_backward = [this, other, out] {
        grad += out->grad;
        other->grad += out->grad;
    };
    return out;
}

std::shared_ptr<Value> Value::operator*(const std::shared_ptr<Value>& other) {
    prev = std::unordered_set<std::shared_ptr<Value>>{shared_from_this(), other};

    auto out = std::make_shared<Value>(data * other->data, prev, "*");

    out->_backward = [this, other, out] {
        grad += other->data * out->data;
        other->grad += data * out->data;
    };
    return out;
}

void Value::backward() {
    std::unordered_set<std::shared_ptr<Value>> visited;
    std::function<void(const std::shared_ptr<Value>&)> build_topo;

    build_topo = [&](const std::shared_ptr<Value>& v) {
        if (visited.find(v) != visited.end())
            return;

        visited.insert(v);

        for (const auto& child : v->prev) {
            build_topo(child);
        }

        v->_backward();
    };

    build_topo(shared_from_this());
}

// Implementation of the non-member operators

std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
    return (*lhs) + rhs;
}

std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs) {
    return (*lhs) * rhs;
}