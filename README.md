# polyglot-micrograd
Inspired by [@karpathy's - micrograd](https://github.com/karpathy/micrograd).
A tiny "polyglot" autograd engine and a neural net library on top of it with PyTorch-like API. 

In C++, Cuda, Rust, Go, Python, and JAX.

This is an educational repository for viewers and the contributors to understand the practical contrasts of a bare-bones autogrid engine and neural net across programming languages.

Programming languages present different coding philosophy and styles. These philosophies lead to significant differences between their performances, like inference speed or memory consumption. 
In the world of data intensive Deep Learning, optimizing each bit operation can become imperative. 
Hence, this is an effort to understand the pros and cons of different languages conditioned on a simple autogrid engine.

Quite obviously this repo is not for production. It can however, assist anyone in the following.
1. If you want to grasp basic implementation contrasts between prog. languages over a single logic (autograd in this case).
2. If you want to witness the performance contrast b/w prog. languages at small scale.
3. If you want to develop a mental model of an autograd engine, and how different languages bring it to life.
4. If you love micrograd, but want it in different languages.
5. If you want to learn a new language by building projects, and have already understood and fallen in love with micrograd's pythonic implementation (this was my main purpose)

Now since, I am merely a beginner in all the languages in scope of this project except python. My implementations can be very sub-optimal, and even non sensical sometimes.
To make this repository actually useful, it will definitely need contributions from anyone who can make any part better.
So we will have open contributions for anyone interested. Hopefully we can make the repository more legitimate with time.
So feel free to add a PR/issue, we would really appreciate a collaboration! 
