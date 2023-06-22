from engine import Value

v1 = Value(2.5)
v2 = Value(3.7)
v3 = Value(-3.0)
v4 = Value(1.7)

v1_2 = v1+v2
v1_2_3 = v1_2*v3
result = v1_2_3+v4

print(result)
result.backward()

for v in [v1,v2,v3,v4]:
    print(f'grad({v}): v.grad')
    