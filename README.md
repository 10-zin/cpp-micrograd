# cpp-micrograd
Inspired by [@karpathy's - micrograd](https://github.com/karpathy/micrograd).

Autograd engine is the technical implementation of backpropogation algorithm that allows neural nets to learn.
And micrograd is the simplest implementation of the autograd engine, but.. its only in python.

This is a simple re-implementation of micrograd using cpp.
It was made with the major intention for personal learning. So..

### Getting started.
Dive into `cpp-micrograd` to get started.
There's a simple getting started code, to create a basic neural net that models the AND logic gate.

### Who will find this repo useful?
1. If you love micrograd, but would wanna also have a cpp version for it.
2. If you want a crisp backprop theory and annotated code of the autograd engine. 
3. If you wanna learn cpp by building neural nets, then this could be a good start (it was my purpose).

### Notable insightful material:
1. `digin-micrograd-theory` consists of fundamental "to-the-point" theory behind autograd. It's based on Karpathy's explanation, customized for getting started with this repository.
2. The entire code-base is annotated with comments, to make the code readable and educational. So do read the docstrings along.

### Vision
Given the recent breakthrough of C/C++ versions of neural nets, like gerganov's llama.cpp, it made a lot of sense to build some neural nets with C/C++, hence cpp-micrograd.

Albeit a toy version, it gives a good understanding of how c++ would implement basic neural nets . IMO a very good start to understanding and using c/c++ neural nets like ggml, because no matter how complex and versatile the network, the basic autograd computation graph will always be same and omnipresent.

**The vision** is to make a C implementation too, and then go all the way to launching Cuda kernels, while making it as educational as possible. 

### Contributions
I am also a novice cpp programmer, so my implementations can be very sub-optimal.
To make this repository actually useful, it will definitely need contributions from anyone who can make any part better.
So we will have open contributions for anyone interested.
So feel free to add a PR/issue, it is free-form for now.

### Credits
1. [@karpathy](https://github.com/karpathy) for the perfect [NeuralNets](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) course
2. [ChatGPT](https://chat.openai.com/) for being the perfect co-pilot!



<!-- I implemented cpp-micrograd a C++ version of Karpathy's autograd engine.

https://github.com/10-zin/cpp-micrograd

I have been working on this for almost 3 weeks on and off. Tho I really kicked it off last week once I got the main engine with backward pass working.

Given the recent breakthrough of C/C++ versions of neural nets, like gerganov's llama.cpp, it made a lot of sense to build some neural nets with C/C++, hence cpp-micrograd.

Albeit a toy version, it gives a good understanding of how c++ would implement basic neural nets . IMO a very good start to understanding and using c/c++ neural nets, like ggml as no matter how complex the network, the basic autograd computation graph will always be the core.

The vision is to make a C implementation too, and then go all the way to launching Cuda kernels, while making it as educational as possible. 
Here is what value you can get from the repo currently.
 If you love micrograd, but would wanna also have a cpp version for it. 
If you want a crisp backprop theory and annotated code of the autograd engine.
 If you wanna learn cpp by building neural nets, then this could be a good start (it was my purpose).
So.. if you find it interesting, do support the project with stars, and contributions. Thanks for reading! -->