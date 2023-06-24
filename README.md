# cpp-micrograd
Inspired by [@karpathy's - micrograd](https://github.com/karpathy/micrograd).

Autograd engine is the technical implementation of backpropogation algorithm that allows neural nets to learn.
And micrograd is the simplest implementation of the autograd engine, but.. its only in python.

This is a simple re-implementation of micrograd using cpp.
It was made with the major intention for personal learning. So..

### Who will find this repo useful?
1. If you love micrograd, but would wanna also have a cpp version for it.
2. If you want a crisp backprop theory and annotated code of the autograd engine. 
3. If you wanna learn cpp by building neural nets, then this could be a good start (it was my purpose).

### TODO: Notable insightful material:
1. `cpp-micrograd/digin` will consist of annotated version of the code. It will dive deep into seemingly complex cpp keywords and design. Through that you can get a deeper understanding of cpp and design choices made.
2. `digin-micrograd-theory` will consist of fundamental "to-the-point" theory behind autograd. It's based on Karpathy's explanation, customized for getting started with this repository.

### Vision
Given the recent breakthrough of C/C++ versions of neural nets, like gerganov's llama.cpp, it made a lot of sense to build some neural nets with C/C++, hence cpp-micrograd.

Albeit a toy version, it gives a good understanding of how c++ would implement basic neural nets . IMO a very good start to understanding and using c/c++ neural nets, like ggml as no matter how complex the network, the basic autograd computation graph will always be the core.

**The vision** is to make a C implementation too, and then go all the way to launching Cuda kernels, while making it as educational as possible. 

### Contributions
I am also a novice cpp programmer, so my implementations can be very sub-optimal.
To make this repository actually useful, it will definitely need contributions from anyone who can make any part better.
So we will have open contributions for anyone interested.
So feel free to add a PR/issue, it is free-form for now.

### Credits
1. [@karpathy](https://github.com/karpathy) for the perfect [NeuralNets](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) course
2. [ChatGPT](https://chat.openai.com/) for being the perfect co-pilot!