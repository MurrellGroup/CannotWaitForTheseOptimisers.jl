using MatrixSign
using Optimisers: AbstractRule, @lazy, @..
import Optimisers: init, apply!, adjust!

nonfirstdims(x, dims=ndims(x)) = prod(size(x)[2:dims])
nonfirstdims(x, ::Nothing) = nonfirstdims(x)

"""
    Muon(η = 0.02, μ = 0.95, λ = 0.01; dims = nothing)
    Muon(; [eta, mu, lambda, dims])

Muon - MomentUm Orthogonalized by Newton-schulz (https://github.com/KellerJordan/Muon)

Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-processing step,
in which each 2D parameter's update is replaced with the nearest orthogonal matrix using Newton-Schulz iteration.

# Parameters
- Learning rate (`η == eta`): Amount by which gradients are discounted before updating the weights
- Momentum (`μ == mu`): Controls the acceleration of gradient descent in the prominent direction
- Weight decay (`λ == lambda`): Controls the strength of ``L_2`` regularisation.
- Keyword `dims`: Dimensions to orthogonalize. If `nothing`, then trailing dimensions get flattened
  into the second dimension. If `dims < ndims(x)`, remaining dimensions are orthogonalized independently.

Note: Works best with large batch sizes and may not be suitable for fine-tuning.
In nanoGPT speedrun experiments, Muon is used for the internal layer >2D weights, and AdamW is used for the 1D weights, embeddings, and heads.

`Optimisers.adjust!(optimiser_state, η::Real)` will adjust the fallback optimizer's `eta` to `η * (opt.eta / eta)`, and Muon's `eta` to `η`, preserving their ratio,
but `Optimisers.adjust!(optimiser, eta = η)` will only adjust Muon's learning rate (allowing you to adjust the fallback optimizer's learning rate separately).
"""
@kwdef struct Muon <: AbstractRule
    eta = 0.02
    mu = 0.95
    lambda = 0.01
    dims = nothing
end

init(::Muon, x::AbstractArray) = zero(x)

function apply!(
    (; eta, mu, lambda, dims)::Muon,
    state, x::AbstractArray{T}, dx
) where T
    η, μ, λ = T(eta), T(mu), T(lambda)
    # update momentum
    @.. state = μ * state + (1-μ) * dx
    # Nesterov update fed to msign
    U = @. μ * state + (1-μ) * dx
    # orthogonalize
    Ot = msign!(
        reshape(U, size(U, 1), nonfirstdims(U, dims), :),
        steps=5, fused=3)
    # post shape factor
    s = √max(1, T(size(Ot, 1) / size(Ot, 2)))
    dx′ = @lazy η * (Ot * s + λ * x)   # decoupled WD, step will subtract dx′
    return state, dx′
end

adjust!(r::Muon, η::Real) = adjust!(r, eta = η)
