# C-micrograd

In the works..
Works for small neural nets, and lesser data.

ToDo Improvements :
> Suffers exploding gradients, as data is increased (needs relu, DIV, SUB operators to handle that).
> Cannot load bigger neural nets, params size is hard-coded to be max of 100 XD (needs to be made dynamic).

## Getting started
```
>> gcc -o run_mlp train.c    
>> ./run_mlp
```