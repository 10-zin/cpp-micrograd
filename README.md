# cpp-micrograd
Inspired by [@karpathy's - micrograd](https://github.com/karpathy/micrograd).

Autograd engine is the technical implementation of backpropogation algorithm that allows neural nets to learn.
And micrograd is the simplest implementation of the autograd engine, but.. its only in python.

This is a simple re-implementation of micrograd using cpp.
It was made with the major intention for personal learning. So..

### Who will find this repo useful?
1. If you want a crisp backprop theory and annotated code of the autograd engine.
2. If you love micrograd, but would wanna see one cpp version for it.
3. If you wanna learn cpp by building neural nets, then this could be a good start (it was my purpose).

### Notable insightful material:
1. I have annotated any seemingly complex keyword/design choices in code with their reason in `cpp-micrograd/digin`.
Through that you can get a deeper understanding of cpp and implementation details of neural nets.
2. I have also added few theoretical snippets, crisply show-casing how and why backprop via autograd engine works. You should check that out first in `digin-micrograd-theory`.

### Contributions
I am also a novice cpp programmer, so my implementations can be very sub-optimal.
To make this repository actually useful, it will definitely need contributions from anyone who can make any part better.
So we will have open contributions for anyone interested.
So feel free to add a PR/issue, we would really appreciate a collaboration! 

### Credits
1. [@karpathy](https://github.com/karpathy) for the perfect [NeuralNets](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) course
2. [ChatGPT](https://chat.openai.com/) for being the perfect co-pilot!