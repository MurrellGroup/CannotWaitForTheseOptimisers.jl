###
#adjust with type control
###

#This would go in adjust.jl
adjust!(ℓ::Leaf, oT::Type, eta::Real) = (ℓ.rule = adjust(ℓ.rule, oT, eta); nothing)
adjust!(ℓ::Leaf, oT::Type; kw...) = (ℓ.rule = adjust(ℓ.rule, oT; kw...); nothing)

adjust(ℓ::Leaf, oT::Type, eta::Real) = Leaf(adjust(ℓ.rule, oT, eta), ℓ.state, ℓ.frozen)
adjust(ℓ::Leaf, oT::Type; kw...) = Leaf(adjust(ℓ.rule, oT; kw...), ℓ.state, ℓ.frozen)

adjust!(tree, oT::Type, eta::Real) = foreach(st -> adjust!(st, oT, eta), tree)
adjust!(tree, oT::Type; kw...) = foreach(st -> adjust!(st, oT; kw...), tree)

adjust(r::AbstractRule, oT::Type, eta::Real) = ifelse(isa(r, oT), adjust(r, eta), r)
adjust(r::AbstractRule, oT::Type; kw...) = ifelse(isa(r, oT), adjust(r; kw...), r)

adjust!(r::AbstractRule, oT::Type, eta::Real) = ifelse(isa(r, oT), adjust!(r, eta), r)
adjust!(r::AbstractRule, oT::Type; kw...) = ifelse(isa(r, oT), adjust!(r; kw...), r)

function adjust(tree, oT::Type, eta::Real)
  t′ = fmap(copy, tree; exclude = maywrite)
  adjust!(t′, oT, eta)
  t′
end

function adjust(tree, oT::Type; kw...)
  t′ = fmap(copy, tree; exclude = maywrite)
  adjust!(t′, oT; kw...)
  t′
end

#And this would go in rules.jl after the OptimiserChain definition
adjust(ℓ::OptimiserChain, oT::Type, eta::Real) = OptimiserChain(map(opt -> adjust(opt, oT, eta), ℓ.opts)...)
adjust(ℓ::OptimiserChain, oT::Type; kw...) = OptimiserChain(map(opt -> adjust(opt, oT; kw...), ℓ.opts)...)