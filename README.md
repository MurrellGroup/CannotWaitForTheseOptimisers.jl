# CannotWaitForTheseOptimisers

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://MurrellGroup.github.io/CannotWaitForTheseOptimisers.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://MurrellGroup.github.io/CannotWaitForTheseOptimisers.jl/dev/)
[![Build Status](https://github.com/MurrellGroup/CannotWaitForTheseOptimisers.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/CannotWaitForTheseOptimisers.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/CannotWaitForTheseOptimisers.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/CannotWaitForTheseOptimisers.jl)

A collection of experimental optimizers implemented according to the [Optimisers.jl](https://github.com/FluxML/Optimisers.jl) interface. This package serves as a testing ground for new optimization algorithms before they are potentially incorporated into the main Optimisers.jl package.

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

This package includes recent and experimental optimizers that have not yet been incorporated into [Optimisers.jl](https://github.com/FluxML/Optimisers.jl). All optimizers adhere to the same interface, ensuring seamless integration and compatibility with existing workflows.