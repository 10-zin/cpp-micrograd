# C-micrograd

In the works..
Works for small neural nets, and v.small data.

ToDo Improvements :
1. Suffers exploding gradients, as data is increased 
    a. needs relu, DIV, SUB operators -> DONE
    b. needs softmax to make loss signal relevant (234234.98-1.0 vs 0.688-1.0) -> IN PROGRESS.
2. Cannot load bigger neural nets, params size is hard-coded to be max of 100 XD (needs to be made dynamic). -> IN PROGRESS.

## Getting started
```
>> gcc -o run_mlp train.c    
>> ./run_mlp
```