# Efficient scaling and squaring method for the matrix exponential

[![DOI](https://zenodo.org/badge/791998477.svg)](https://zenodo.org/doi/10.5281/zenodo.11071264)

This repository contains a Julia package to adaptively and efficiently compute matrix exponential up to a given tolerance.

> This work presents a new algorithm to compute the matrix exponential within a given tolerance. Combined with the scaling and squaring procedure, the algorithm incorporates Taylor, partitioned and classical Padé methods shown to be superior in performance to the approximants used in state-of-the-art software. The algorithm computes matrix--matrix products and also matrix inverses, but it can be implemented to avoid the computation of inverses, making it convenient for some problems. If the matrix A belongs to a Lie algebra, then exp(A) belongs to its associated Lie group, being a property which is preserved by diagonal Padé approximants, and the algorithm has another option to use only these. Numerical experiments show the superior performance with respect to state-of-the-art implementations. 

## Installation
```shell
julia

julia> ]
(@v1.10) pkg> add https://github.com/nakopylov/AdaptiveExp
```

## Usage
```julia
using AdaptiveExp

A = rand(5, 5);
B = expadapt(A, 1e-11)
C = expadapt(A) # the same as expadapt(A, 1e-12)
```

## Running examples
First, change the working directory to `AdaptiveExp/examples`, for example, if you are currently in the package's directory:
```shell
cd ./examples
```
Then in Julia activate the (separate) environment in the `examples` directory of the package:
```shell
julia

julia> ]
(@v1.10) pkg> activate .
  Activating project at `~/AdaptiveExp/examples`

(examples) pkg>
```
Run an example in Julia REPL:
```shell
julia> include("./experiment_unit_err_vs_norm.jl")
```