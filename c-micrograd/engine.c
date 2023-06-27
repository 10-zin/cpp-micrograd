#include <stdio.h>
#include <stdlib.h>

typedef struct Value {
    float data;
    float grad;
    void (*_backward)(struct Value*);
    struct Value* prev[2];
    char* op;
} Value;

Value* create_value(float data);

Value* add(Value* self, Value* other);


Value* create_value(float data) {
    Value* value = (Value*)malloc(sizeof(Value));
    value->data = data;
    value->grad = 0;
    value->_backward = NULL;
    value->prev[0] = NULL;
    value->prev[1] = NULL;
    value->op = "";
    return value;
}

Value* add(Value* self, Value* other) {
    Value* out = create_value(self->data + other->data);
    out->prev[0] = self;
    out->prev[1] = other;
    out->op = "+";

    void _backward(Value* self) {
        self->prev[0]->grad += self->grad;
        self->prev[1]->grad += self->grad;
    }

    out->_backward = _backward;

    return out;
}

int main() {
    Value* a = create_value(2.0);
    Value* b = create_value(3.0);

    Value* result = add(a, b);

    printf("a->grad: %f\n", a->grad);
    printf("b->grad: %f\n", b->grad);
    printf("result: %f\n", result->data);

    free(a);
    free(b);
    free(result);

    return 0;
}
