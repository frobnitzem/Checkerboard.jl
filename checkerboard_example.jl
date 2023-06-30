using Checkerboard
using LatticeUtilities

# size of square lattice
function setup(L)
	# define square lattice unit cell
	square  = UnitCell(
		lattice_vecs = [[1.,0.],[0.,1.]],
		basis_vecs = [[0.,0.]]
	)

	# define L×L unit cell lattice with periodic boundary conditions
	lattice = Lattice(
		L = [L,L],
		periodic = [true,true]
	)

	# define neasrest-neighbor bond in x direction
	bond_x  = Bond(
		orbitals = (1,1),
		displacement = [1,0]
	)

	# define nearest-neighbor bond in y direction
	bond_y  = Bond(
		orbitals = (1,1),
		displacement = [0,1]
	)

	# get the number of size in the lattice, i.e. N = L×L for square lattice
	N = nsites(square, lattice)

	# build the neighbor table for an L×L square lattice with periodic boundary condtions
	# and just nearest-neighbor bonds
	neighbor_table = build_neighbor_table([bond_x,bond_y], square, lattice)

	# define uniform hopping amplitude/energy for corresponding square lattice tight-binding model
	t = ones(size(neighbor_table,2))

	# define discretization in imaginary time i.e. the small parameter the in checkerboard approximation
	Δτ = 0.1

	# construct/calculate checkerboard approximation
	return N, CheckerboardMatrix(neighbor_table, t, Δτ)
end

function main(args::Array{String,1})::Int32
	# must initialize scalars
    L::Int32 = 16
    steps::Int32 = 1

    @show args

    # args don't include Julia executable and program
    nargs = size(args)[1]

    if nargs == 2
		L = parse(Int32, args[1])
        steps = parse(Int32, args[2])
    else
        throw( ArgumentError(string("Usage: ", argv[0], " <L> <steps>")) )
    end
	if steps < 1
		throw( ArgumentError("Steps must be at least 2") )
	end

	# (Higher Priority)
	# For the hybrid/hamiltonian monte carlo method we need to be able to multiply the a vector by the
	# checkerboard approximation. Below, we use directly call the low-level/developer api methhods rather
	# than the high-level public facing API that most people would use when calling this package.

	# let us unpack the elements of the CheckerboardMatrix type, so that we may call the
	# low-level/developer API directly
	(N, (; neighbor_table, coshΔτt, sinhΔτt, colors, transposed, inverted)) = setup(L)


	# define a random vector of length N
	v = randn(N)

	# Perform in-place matrix-vector multipy Γ⋅v.
	# This is the method we would like to port to a GPU.
	timings = zeros(steps)
	for i = 1:steps
		timings[i] = @elapsed Checkerboard.checkerboard_lmul!(v, neighbor_table, coshΔτt, sinhΔτt, colors,
							transposed = transposed, inverted = inverted)
	end
	average_time = sum(timings[2:steps]) / (steps - 1)
	println("Size ", N, ", time per step (us) ", average_time*1e6)

	# (Lower Priority)
	# In DQMC we instead need to perform matrix by checkerboard approximation.
	# Here is an example of the low-level/developer API call to do this

	# initialize random matrix
	#M = randn(N,N)

	# Perform in-place matrix-matrix multiply Γ⋅M.
	# This is the method we would like to port to a GPU.
	#@time Checkerboard.checkerboard_lmul!(M, neighbor_table, coshΔτt, sinhΔτt, colors,
	#								transposed = transposed, inverted = inverted)

	return 0
end

main(ARGS)
