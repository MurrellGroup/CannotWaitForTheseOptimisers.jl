module CannotWaitForTheseOptimisers

using Optimisers, Random
import Optimisers: OptimiserChain, AbstractRule, Leaf, adjust, adjust!, _adjust, AdamW, _norm, norm, zero, apply!, init, fmap, maywrite, @.., @lazy

include("rules.jl")
include("adjust.jl")

export Muon, Apollo, NormGrowthCap, GradNormControl, AdaptiveGradNormControl

end
