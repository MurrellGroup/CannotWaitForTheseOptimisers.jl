module Experimental

include("Muon.jl")
export Muon

using Base: IdSet
export IdSet

using Functors: fcollect
findnodes(pred::Function, x) = filter(pred, fcollect(x))
export findnodes

end
