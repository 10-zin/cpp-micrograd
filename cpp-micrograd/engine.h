#ifndef SCRATCH_ENGINE_H
#define SCRATCH_ENGINE_H

#include <functional>
#include <unordered_set>
#include <string>
#include <memory>

class Value;

std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs);
std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& lhs, const std::shared_ptr<Value>& rhs);

class Value : public std::enable_shared_from_this<Value> {
private:
    double data;
    double grad;
    std::function<void()> _backward;
    std::unordered_set<std::shared_ptr<Value>> prev;
    std::string op;

public:
    Value(double data, std::unordered_set<std::shared_ptr<Value>> prev = {}, std::string op = "");

    double get_data() const;
    void zero_grad();

    std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& other);
    std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& other);

    void backward();
};

#endif  // SCRATCH_ENGINE_H
