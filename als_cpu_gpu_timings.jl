### A Pluto.jl notebook ###
# v0.20.6

using Markdown
using InteractiveUtils

# ╔═╡ 51cebcb8-23b8-11f1-263d-19571a46f9f5
using Pkg; Pkg.activate(@__DIR__)

# ╔═╡ b59d43f3-d46c-4ecf-8005-32cd4d979c94
using BenchmarkTools

# ╔═╡ 3c061f37-6382-40f1-a796-17ac5837f60d
using CairoMakie

# ╔═╡ dc18f2fe-470f-44e2-8601-24403f888ace
using GCPDecompositions

# ╔═╡ 00093a4f-c3b4-4588-97b7-2ad22dcf23e7
using Metal

# ╔═╡ 342aff08-4b28-4409-96fc-f323a8b692e4
using Random

# ╔═╡ d2027f0b-fb2e-46b7-9ba1-085259ed80ef
md"""
### Setup
"""

# ╔═╡ 16b09790-ffa6-407b-b17d-94a5ca8017ab
sizes = [10, 25, 50, 100, 150, 200, 250, 300, 400, 500, 600, 800, 1000]

# ╔═╡ 2d414bc3-18bd-43e2-afe4-c362d8902177
r = 5;

# ╔═╡ b03dc8b2-bcb8-48a1-ba4e-a18f4041cbf2
md"""
### Run Experiment
"""

# ╔═╡ fd8d60ac-e9a4-449e-b62b-3343ccfe55db
function time_cpu(sz)
	
	# Ensure that inits are same for cpu and gpu tests
	# (although data for CPU has to be Float64).
	# randn seems to have an issue with the metal rng where it will produce NaNs with some probability, so need to use standard rng
	rng = Random.default_rng()
	metal_rng = MPS.RNG()
	Random.seed!(rng, sz)
	Random.seed!(metal_rng, sz)
	X = Random.randn(rng,sz,sz,sz)
	X_gpu = MtlArray(convert(Array{Float32},X))

	loss = LeastSquaresLoss()
	constraints = default_gcp_constraints(X, r, loss)
	algorithm = CP_ALS()
	M_init_gpu = default_gcp_init(metal_rng, X_gpu, r, loss, constraints, algorithm)
	M_init = CPD(convert(Array{Float64}, Array(M_init_gpu.λ)), Array.(convert.(Array{Float64}, M_init_gpu.U)))
	
	# Deallocate
	X_gpu = nothing
	M_init_gpu = nothing

	# Run
	gcp(X, r; init=M_init, algorithm=algorithm)  # Compile once
	return @benchmark gcp($X,$r; init=$M_init, algorithm=$algorithm)  # Time
end

# ╔═╡ bad7218b-ac94-428c-bac7-6c9473aa9b0c
function time_gpu(sz)

	# Ensure that inits are same for cpu and gpu tests
	# (although data for CPU has to be Float64).
	# randn seems to have an issue with the metal rng where it will produce NaNs with some probability, so need to use standard rng
	rng = Random.default_rng()
	metal_rng = MPS.RNG()
	Random.seed!(rng, sz)
	Random.seed!(metal_rng, sz)
	X = Random.randn(rng,sz,sz,sz)
	X_gpu = MtlArray(convert(Array{Float32},X))
	
	loss = LeastSquaresLoss()
	constraints = default_gcp_constraints(X, r, loss)
	algorithm = CP_ALS()
	M_init_gpu = default_gcp_init(metal_rng, X_gpu, r, loss, constraints, algorithm)

	# Deallocate
	X = nothing
	
	gcp(X_gpu, r; init=M_init_gpu, algorithm=algorithm)  # Compile once
	return @benchmark Metal.@sync gcp($X_gpu, $r; init=$M_init_gpu, algorithm=$algorithm)  # Time
end

# ╔═╡ 5b1ec5ac-7407-4861-838e-462c019d1ee3
cpu_results = [time_cpu(sz) for sz in sizes]

# ╔═╡ 345b801e-5b8e-4d65-b637-91670fbbbfb6
gpu_results = [time_gpu(sz) for sz in sizes]

# ╔═╡ 813ef113-2d76-4865-b98d-718ef7d4f079
md"""
### Plot Results 
"""

# ╔═╡ 2959e451-cb41-4e78-83b0-ba16432e6ac7
results_plot = let
	
	fig = Figure(; size=(700, 400))
	ax = Axis(fig[1,1]; xlabel="Tensor size (n x n x n)", ylabel="Runtime (s)")
	
	cpu_times = [mean(res.times)/10^6 for res in cpu_results]
	gpu_times = [mean(res.times)/10^6 for res in gpu_results]
	
	lines!(ax, sizes, cpu_times, color=:red, label="CPU")
	scatter!(ax, sizes, cpu_times, color=:red)

	lines!(ax, sizes, gpu_times, color=:blue, label="Metal")
	scatter!(ax, sizes, gpu_times, color=:blue)
	
	Legend(fig[1,2], ax)
	
	fig
end

# ╔═╡ dd0f055c-27a4-45b8-a0d9-6d4c9aef3c2f


# ╔═╡ 5db7d269-1298-4a5a-82bb-792270b62423


# ╔═╡ 2f5838df-9b80-49f6-a210-0b7f7c2679be


# ╔═╡ Cell order:
# ╠═51cebcb8-23b8-11f1-263d-19571a46f9f5
# ╠═b59d43f3-d46c-4ecf-8005-32cd4d979c94
# ╠═3c061f37-6382-40f1-a796-17ac5837f60d
# ╠═dc18f2fe-470f-44e2-8601-24403f888ace
# ╠═00093a4f-c3b4-4588-97b7-2ad22dcf23e7
# ╠═342aff08-4b28-4409-96fc-f323a8b692e4
# ╟─d2027f0b-fb2e-46b7-9ba1-085259ed80ef
# ╠═16b09790-ffa6-407b-b17d-94a5ca8017ab
# ╠═2d414bc3-18bd-43e2-afe4-c362d8902177
# ╟─b03dc8b2-bcb8-48a1-ba4e-a18f4041cbf2
# ╠═fd8d60ac-e9a4-449e-b62b-3343ccfe55db
# ╠═bad7218b-ac94-428c-bac7-6c9473aa9b0c
# ╠═5b1ec5ac-7407-4861-838e-462c019d1ee3
# ╠═345b801e-5b8e-4d65-b637-91670fbbbfb6
# ╟─813ef113-2d76-4865-b98d-718ef7d4f079
# ╠═2959e451-cb41-4e78-83b0-ba16432e6ac7
# ╠═dd0f055c-27a4-45b8-a0d9-6d4c9aef3c2f
# ╠═5db7d269-1298-4a5a-82bb-792270b62423
# ╠═2f5838df-9b80-49f6-a210-0b7f7c2679be
