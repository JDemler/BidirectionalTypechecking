# Implementation of "Complete and Easy Bidirectional Typechecking for Higher-Rank Polymorphism"
See [arXiv:1306.6032](https://arxiv.org/abs/1306.6032)

This implementation focusses on readability and being able to follow the paper while reading the code.

Use the `synth` function to synthesize a Type from an Expression.
The process is traced through stdout, so that you can follow the rules invoked.

Basis of this implementation was the paper and implementations by [Olle Fredriksson](https://github.com/ollef/Bidirectional) in Haskell and [Albert ten Napel](https://github.com/atennapel/bidirectional.js) in TypeScript.
