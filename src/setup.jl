# See https://github.com/FluxML/Optimisers.jl/pull/204

using Optimisers: isnumeric, _trainable, mapvalue

setup(rule::AbstractRule, model) = setup(Returns(rule), model)
function setup(fun::Function, model)
  cache = IdDict()
  tree = _setup(fun, model; cache)
  isempty(cache) && @warn "setup found no trainable parameters in this model"
  tree
end

# _setup is almost fmapstructure, but needs a _trainable_walk, and a cache which ignores numbers etc.
function _setup(fun::Function, x; cache)
  haskey(cache, x) && return cache[x]
  if isnumeric(x)
    rule = fun(x)::AbstractRule
    ℓ = Leaf(rule, init(rule, x))
    if isbits(x)
      cache[nothing] = nothing  # just to disable the warning
      ℓ
    else
      cache[x] = ℓ
    end
  else
    mapvalue(xᵢ -> _setup(fun, xᵢ; cache), _trainable(x))
  end
end