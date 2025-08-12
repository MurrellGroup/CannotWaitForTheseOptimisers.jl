
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
      η = T(o.eta); μ = T(o.mu); λ = T(o.lambda)
      # momentum: m ← β m + (1-β) g
      @.. state = μ * state + (one(T) - μ) * dx
      # Nesterov update fed to NS5: U ← (1-β) g + β m
      U = @.. (one(T) - μ) * dx + μ * state
      # orthogonalize + post shape factor √max(1, r/c)
      Ot = _newton_schulz5(U)
      r = size(x, 1); c = nonfirstdims(x)
      s = T(sqrt(max(one(T), T(r) / T(c))))
      dx′ = @lazy η * (Ot * s + λ * x)   # decoupled WD, step will subtract dx′
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
    X = G / (norm(G) + T(1e-7))
    if size(G, 1) > size(G, 2)
      return transpose(_inner_newton_schulz5(transpose(X)))
    else
      return _inner_newton_schulz5(X)
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
    current_norm = _norm(Optimisers.unthunk(dx), 2)
    if o.throw && !isfinite(current_norm)
        throw(DomainError("gradient has L2-norm $current_norm, for array $(summary(x))"))
    end
    if state == 0
        return (current_norm), dx
    else
        #If you're below the hard min, then don't scale
        if o.scale
            minthresh = o.lb * sqrt(length(Optimisers.unthunk(dx)))
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
    GradNormControl(accumulator, τ = 1.1; epsilon = 1e-8, lb = 0.1, throw = true, scale = true, clipreportthresh = Inf)

NormGrowthCap with additional control, accumulation, and reporting options.
`accumulator` must be an array of `Float64` with two elements, which is where the unscaled and scaled gradient norms are added into, allowing you to monitor the sum of the norms. It is your job to print/reset this.
"""
struct GradNormControl <: Optimisers.AbstractRule
    tau::Float64
    epsilon::Float64
    lb::Float64 #Min grad norm, to stop a tensor getting stuck near zero
    throw::Bool
    scale::Bool
    heavyclipthresh::Real
    accumulator::AbstractVector{<:Float64}
end

function GradNormControl(accumulator, τ = 1.1; epsilon = 1e-8, lb = 0.1, throw = true, scale = true, clipreportthresh = Inf)
    if length(accumulator) != 2
        throw(ArgumentError("accumulator must be an array with two elements, initialized to 0"))
    end
    GradNormControl(τ, epsilon, lb, throw, scale, clipreportthresh, accumulator)
end

function init(o::GradNormControl, x::AbstractArray{T}) where T
    if o.scale
        minthresh = o.lb * sqrt(length(x))
    else
        minthresh = o.lb
    end
    return T(0), minthresh
end

function apply!(o::GradNormControl, state, x::AbstractArray{T}, dx) where T
    prevnorm, minthresh = state
    utdx = Optimisers.unthunk(dx)
    current_norm = Optimisers._norm(utdx, 2)
    o.accumulator[1] += current_norm
    if o.throw && !isfinite(current_norm)
        throw(DomainError("gradient has L2-norm $current_norm, for array $(summary(x))"))
    end
    if prevnorm == 0
        o.accumulator[2] += current_norm
        return (current_norm, minthresh), dx
    else
        if current_norm < minthresh
            o.accumulator[2] += current_norm
            return (current_norm, minthresh), dx
        end
        ratio = current_norm / (prevnorm + o.epsilon)
        if ratio > o.tau
            lambda = T((o.tau * prevnorm) / (current_norm + o.epsilon))
            if ratio > o.tau * o.heavyclipthresh
                println("Heavy clipping on $(size(utdx)):", current_norm, "->", current_norm * lambda)
            end
            o.accumulator[2] += current_norm * lambda
            return (current_norm * lambda, minthresh), dx * lambda
        else
            o.accumulator[2] += current_norm
            return (current_norm, minthresh), dx
        end
    end
end


"""
    AdaptiveGradNormControl(accumulator, τ = 1.0; epsilon = 1e-8, lb = 0.1, 
                           momentum = 0.90, throw = true, clipreportthresh = Inf)

Gradient norm control using exponential moving statistics. Clips gradients when the 
current norm exceeds mean + τ * std.
"""
struct AdaptiveGradNormControl <: Optimisers.AbstractRule
    tau::Float64
    epsilon::Float64
    lb::Float64
    throw::Bool
    momentum::Float64
    heavyclipthresh::Real
    accumulator::AbstractVector{<:Float64}
end

function AdaptiveGradNormControl(accumulator, τ = 1.0; epsilon = 1e-8, lb = 0.1, 
                                momentum = 0.9, throw = true, clipreportthresh = Inf)
    if length(accumulator) != 2
        throw(ArgumentError("accumulator must be an array with two elements"))
    end
    AdaptiveGradNormControl(τ, epsilon, lb, throw, momentum, clipreportthresh, accumulator)
end

# Helper function to update running statistics
function update_running_stats(curr_norm, prev_mean, prev_std, momentum)
    new_mean = momentum * prev_mean + (1 - momentum) * curr_norm
    # Variance update formula: var = E[(x - μ)²] = E[x²] - μ²
    new_var = momentum * (prev_std^2 + prev_mean^2) + 
              (1 - momentum) * curr_norm^2 - new_mean^2
    new_std = sqrt(max(new_var, 1e-8))
    return new_mean, new_std
end

function Optimisers.init(o::AdaptiveGradNormControl, x::AbstractArray{T}) where T
    minthresh = o.lb * sqrt(length(x))
    return (T(0), T(0), minthresh)  # mean, std, minthresh
end

function Optimisers.apply!(o::AdaptiveGradNormControl, state, x::AbstractArray{T}, dx) where T
    mu, std, minthresh = state
    utdx = Optimisers.unthunk(dx)
    current_norm = Optimisers._norm(utdx, 2)
    o.accumulator[1] += current_norm
    if o.throw && !isfinite(current_norm)
        throw(DomainError("gradient has L2-norm $current_norm"))
    end
    if current_norm < minthresh
        o.accumulator[2] += current_norm
        new_mean, new_std = update_running_stats(current_norm, mu, std, o.momentum) #Unsure if we should adjust the mean if they fall below the threshold?
        return (new_mean, new_std, minthresh), dx
    end
    if mu == 0
        o.accumulator[2] += current_norm
        return (current_norm, current_norm, minthresh), dx
    end
    threshold = mu + o.tau * std
    if current_norm > threshold
        lambda = T(threshold / (current_norm + o.epsilon))
        clipped_norm = current_norm * lambda
        if current_norm > threshold * o.heavyclipthresh
            println("Heavy clipping on $(size(utdx)): ", current_norm, "->", clipped_norm, " with mu ", mu, " and std ", std)
        end
        new_mean, new_std = update_running_stats(clipped_norm, mu, std, o.momentum)
        o.accumulator[2] += clipped_norm
        return (new_mean, new_std, minthresh), dx * lambda
    end
    o.accumulator[2] += current_norm
    new_mean, new_std = update_running_stats(current_norm, mu, std, o.momentum)
    return (new_mean, new_std, minthresh), dx
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
