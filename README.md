# CannotWaitForTheseOptimisers

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MurrellGroup.github.io/CannotWaitForTheseOptimisers.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/CannotWaitForTheseOptimisers.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/CannotWaitForTheseOptimisers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/CannotWaitForTheseOptimisers.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/CannotWaitForTheseOptimisers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/CannotWaitForTheseOptimisers.jl)

A collection of experimental optimizers implemented according to the [Optimisers.jl](https://github.com/FluxML/Optimisers.jl) interface. We intend to use this package as a testing ground for new optimization algorithms, and then possibly get them incorporated into the main Optimisers.jl package. As such, please do not expect much stability from this package.

## Installation

  ```julia
  pkg> add Optimisers
  pkg> add https://github.com/MurrellGroup/CannotWaitForTheseOptimisers.jl
  ```

## Usage

  ```julia
  using CannotWaitForTheseOptimisers, Optimisers
  ```

## Description

This package currently includes attempts at implementing:

- [x] [Muon](https://kellerjordan.github.io/posts/muon/) which performs an orthogonalization step before parameter update, and seems excellent for training transformers.
- [x] [Apollo](https://arxiv.org/abs/2412.05270) which tracks low rank moments using a random projection, reducing the memory footprint of the optimizer.
- [x] [NormGrowthCap](https://arxiv.org/abs/2410.01623) which prevents the norm of the parameters from growing too quickly.
