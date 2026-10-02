## Tensor Kernels

"""
Tensor kernels for Generalized CP Decomposition.
"""
module TensorKernels

using ..GCPDecompositions
using Compat: allequal
using LinearAlgebra: mul!, rmul!
using SparseArrays: AbstractSparseMatrix, sparse
using SparseArrayKit: SparseArray, nonzero_length, nonzero_keys, nonzero_values
using Base.Cartesian: @nloops, @ntuple, @nexprs, @nref
using StaticArrays: MVector
import Combinatorics
#using SparseTensors: AbstractSparseTensor, numstored, storedindices, storedvalues
export create_mttkrp_buffer, mttkrp, mttkrp!, mttkrps, mttkrps!, khatrirao, khatrirao!
export sparse_mttkrp!, sparse_mttkrps!
export checksym
export symmetrize_tensor, collect_multinomial_coefficients
export fill_reduced_Y_vec!, columnwise_ttv_all_modes!, columnwise_ttv_all_modes_except_one!
export ttv_threading_plan, fill_reduced_Y_vec_multithreaded!
export columnwise_ttv_all_modes_multithread!, columnwise_ttv_all_modes_except_one_multithread!
    
include("tensor-kernels/khatrirao.jl")
include("tensor-kernels/mttkrp.jl")
include("tensor-kernels/mttkrps.jl")
include("tensor-kernels/symmetric_kernels.jl")
include("tensor-kernels/checksym.jl")

end
