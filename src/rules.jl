
nonfirstdims(x) = prod(size(x)[2:end])

"""
    Muon(opt = AdamW(eta = 0.0003, beta = (0.9,0.95), lambda = 0.01), η = 0.02, μ = 0.95, λ = 0.01, fallback = Returns(false))
    Muon(; [opt, eta, mu, lambda, fallback])

Muon - MomentUm Orthogonalized by Newton-schulz (https://github.com/KellerJordan/Muon)

Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-processing step,
in which each 2D parameter's update is replaced with the nearest orthogonal matrix using Newton-Schulz iteration.

# Parameters
- Fallback optimizer (`opt`): Optimizer to use for 1D parameters or when the `fallback` function returns true
- Learning rate (`η == eta`): Amount by which gradients are discounted before updating the weights
- Momentum (`μ == mu`): Controls the acceleration of gradient descent in the prominent direction
- Weight decay (`λ == lambda`): Controls the strength of ``L_2`` regularisation.
- Fallback function (`fallback`): Function to control when, in addition to 1D arrays, the fallback optimizer should be used. Will be passed the parameter array and must return a boolean.

Note: Works best with large batch sizes and may not be suitable for fine-tuning.
In nanoGPT speedrun experiments, Muon is used for the internal layer >2D weights, and AdamW is used for the 1D weights, embeddings, and heads.

`Optimisers.adjust!(optimiser_state, η::Real)` will adjust the fallback optimizer's `eta` to `η * (opt.eta / eta)`, and Muon's `eta` to `η`, preserving their ratio,
but `Optimisers.adjust!(optimiser, eta = η)` will only adjust Muon's learning rate (allowing you to adjust the fallback optimizer's learning rate separately).
"""
struct Muon <: AbstractRule
    opt::AbstractRule
    eta::Float64
    mu::Float64
    lambda::Float64
    fallback::Function
end

Muon(;opt = AdamW(eta = 0.0003, beta = (0.9,0.95), lambda = 0.01), eta = 0.02, mu = 0.95, lambda = 0.01, fallback = x -> false) = Muon(opt, eta, mu, lambda, fallback)

function init(o::Muon, x::AbstractArray)
  if nonfirstdims(x) == 1 || o.fallback(x)
    return init(o.opt, x)
  else
    return zero(x)
  end
end

function apply!(o::Muon, state, x::AbstractArray{T}, dx) where T
  if nonfirstdims(x) == 1 || o.fallback(x)
    return apply!(o.opt, state, x, dx)
  else
    η, μ, λ = T(o.eta), T(o.mu), T(o.lambda)
    @.. state = μ * state + dx
    Ot = _newton_schulz5(μ .* state .+ dx) * T(sqrt(max(1, size(x,1)/nonfirstdims(x))))
    dx′ = @lazy η * (Ot + λ * x)
    return state, dx′
  end
end

function _inner_newton_schulz5(X::AbstractMatrix{T}) where T
  a, b, c = (T(3.4445f0), T(-4.7750f0), T(2.0315f0))
  for _ in 1:5
    A = X * X'
    B = b * A + c * A * A
    X = a * X + B * X
  end 
  X
end
function _newton_schulz5(G::AbstractMatrix{T}) where T
    X = G / (norm(G) + eps(T))
    if size(G, 1) > size(G, 2)
      transpose(_inner_newton_schulz5(transpose(X)))
    else
      _inner_newton_schulz5(X)
    end
end
_newton_schulz5(G::AbstractArray) = reshape(_newton_schulz5(reshape(G, size(G,1), :)), size(G))

adjust(r::Muon, η::Real) = adjust(r, eta = η, opt = adjust(r.opt, eta = (r.opt.eta / r.eta) * η))

"""
    NormGrowthCap(τ = 1.01; ϵ = 1e-8, lb = 1e-7, throw = true, scale = true)

Gradient norm growth limiter. `τ` controls the maximum that the gradient norm can grow from one step to the next, such that
if `||dx||/||dx_prev|| > τ` & `||dx|| > lb`, then `dx = dx * τ*||dx_prev||/(||dx||+ϵ)`
Inspired by [Chen et al.](https://arxiv.org/abs/2410.01623) and used with Apollo in [Zhu et al.](https://arxiv.org/abs/2412.05270), but
with Optimisers.jl this will apply per-tensor instead of per-model. This implementation also introduces `lb` as a hard minimum on the gradient norm threshold,
and never rescales grads below this, preventing a tensor from getting "trapped" near zero. This can be a fixed min, or scaled by the square root of the
number of parameters in the tensor (with `scale = true`).
"""
struct NormGrowthCap <: AbstractRule
    tau::Float64
    epsilon::Float64
    lb::Float64 #Min grad norm, to stop a tensor getting stuck near zero
    throw::Bool
    scale::Bool
end

NormGrowthCap(τ = 1.01; ϵ = 1e-8, lb = 1e-7, throw = true, scale = true) = NormGrowthCap(τ, ϵ, lb, throw, scale)

init(o::NormGrowthCap, x::AbstractArray{T}) where T = T(0)

function apply!(o::NormGrowthCap, state, x::AbstractArray{T}, dx) where T
    current_norm = _norm(dx, 2)
    if o.throw && !isfinite(current_norm)
        throw(DomainError("gradient has L2-norm $current_norm, for array $(summary(x))"))
    end
    if state == 0
        return (current_norm), dx
    else
        #If you're below the hard min, then don't scale
        if o.scale
            minthresh = o.lb * sqrt(length(dx))
        else
            minthresh = o.lb
        end
        if current_norm < minthresh
            return current_norm, dx
        end
        ratio = current_norm / (state + o.epsilon)
        if ratio > o.tau
            lambda = T((o.tau * state) / (current_norm + o.epsilon))
            return current_norm * lambda, dx * lambda
        else
            return current_norm, dx
        end
    end
end

"""
    Apollo(opt::AdamW = AdamW(), r::Function = dim -> ceil(Int, sqrt(dim)); u = 100, sort_dims = true)
    Apollo(η::Real, args...; kw...)
    Apollo(arg, rank::Int; kw...)
    Apollo(η::Real, rank::Int; kw...)

Apollo optimizer from Zhu et al. (https://arxiv.org/abs/2412.05270). Tracks moments in a low-rank subspace, aiming for Adam-like behavior with minimal additional memory usage.
First argument can be an AdamW optimizer, or a learning rate (which will use the default AdamW optimizer with that learning rate). Second argument can be a rank, or a function
to compute the rank from the second dimension (or the product of all dims > 1) of the weight matrix (or tensor).
"""
struct Apollo{T1, T2} <: AbstractRule
    opt::T1
    r::T2 #Maps non-first dims to rank
    u::Int #Subspace update frequency (T in paper)
    sort_dims::Bool #Whether to swap the dims of x and dx when the second dim is smaller than the first
end

function adjust(r::Apollo; kw...)
  if (:u in keys(kw)) || (:r in keys(kw)) || (:sort_dims in keys(kw))
    @error "Apollo does not support adjusting: u, r, sort_dims"
  end
  return Apollo(_adjust(r.opt, NamedTuple(kw)), r.r, r.u, r.sort_dims)
end
adjust(r::Apollo, η::Real) = Apollo(adjust(r.opt, η), r.r, r.u, r.sort_dims)


Apollo(opt::AdamW = AdamW(), r::Function = dim -> ceil(Int, sqrt(dim)); u = 100, sort_dims = true) = Apollo(opt, r, u, sort_dims)
Apollo(η::Real, args...; kw...) = Apollo(AdamW(η), args...; kw...)
Apollo(arg, rank::Int; kw...) = Apollo(arg, dim -> min(dim, rank); kw...)
Apollo(η::Real, rank::Int; kw...) = Apollo(AdamW(η), rank; kw...)


#Use the base init and apply for 1D arrays
init(o::Apollo, x::AbstractArray{T,1}) where T = init(o.opt, x)
apply!(o::Apollo, state, x::AbstractArray{T,1}, dx) where T = apply!(o.opt, state, x, dx)

function init(o::Apollo, x::AbstractArray{T}) where T
    first_dim, second_dim = size(x,1), nonfirstdims(x)
    if o.sort_dims && second_dim < first_dim
        first_dim, second_dim = second_dim, first_dim
    end
    rank = o.r(second_dim)
    P = similar(x, rank, first_dim)
    randn!(P)
    P .*= T(sqrt(1/rank))
    ((similar(x, rank, second_dim) .= 0, similar(x, rank, second_dim) .= 0, o.opt.beta), 1, P)
end


function apply!(o::Apollo, state, x::AbstractArray{T}, dx) where T
  swapped = false
  original_size = size(x)
  x = reshape(x, size(x,1), nonfirstdims(x))
  
  dx = Broadcast.materialize(dx) #This is to stop the "gradient type" @lazy test from failing due to reshape.
  dx = reshape(dx, size(x,1), nonfirstdims(x))

  first_dim, second_dim = size(x,1), size(x,2)
  if o.sort_dims && second_dim < first_dim
      first_dim, second_dim = second_dim, first_dim
      x = x'
      dx = dx'
      swapped = true
  end
  (mt, vt, βt), t, P = state
  η = T(o.opt.eta) #This is what will get modified by adjust
  λ = T(o.opt.lambda)
  β = T.(o.opt.beta)
  ϵ = T(o.opt.epsilon)
  βt = T.(βt)
  if mod(t, o.u) == 0 
      rank = o.r(second_dim)
      randn!(P)
      P .*= T(sqrt(1/rank))
  end
  R = P * dx
  @.. mt = β[1] * mt + (1 - β[1]) * R
  @.. vt = β[2] * vt + (1 - β[2]) * abs2(R)
  Rhat = @. mt / (1 - βt[1]) / (sqrt(vt / (1 - βt[2])) + ϵ)

  R2sum = sum(abs2, R; dims=1) 
  Rhat2sum = sum(abs2, Rhat; dims=1)
  s = @. sqrt(Rhat2sum) / (sqrt(R2sum) + ϵ)
  dx′′ = η * (dx .* s) + λ * x 

  if swapped
      dx′′ = transpose(dx′′)
  end
  return ((mt, vt, βt .* β), t+1, P), reshape(dx′′, original_size)
end