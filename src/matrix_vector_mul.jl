####################################
## MATRIX-VECTOR MULTIPLY METHODS ##
####################################

import SparseArrays

@doc raw"""
    checkerboard_lmul!(v::AbstractVector{T}, neighbor_table::Matrix{Int},
        coshΔτt::AbstractVector{E}, sinhΔτt::AbstractVector{E}, colors::Matrix{Int};
        transposed::Bool=false, inverted::Bool=false) where {T<:Continuous, E<:Continuous}

Multiply in-place the vector `v` by the checkerboard matrix.
"""
function checkerboard_lmul!(v::AbstractVector{T}, neighbor_table::Matrix{Int},
    coshΔτt::AbstractVector{E}, sinhΔτt::AbstractVector{E},
    colors::Matrix{Int}; transposed::Bool=false, inverted::Bool=false) where {T<:Continuous, E<:Continuous}

    @assert !(T<:Real && E<:Complex) "Cannot multiply a real valued vector by a complex checkerboard matrix!"

    # number of checkerboard colors
    Ncolors = size(colors, 2)

    # how to iterate over neighbors in neighbor_table accounting for whether
    # or not the checkerboard matrix has been transposed
    transposed = inverted*(1-transposed) + (1-inverted)*transposed
    start      = (1-transposed) + transposed*Ncolors
    step       = 1 - 2*transposed
    stop       = (1-transposed)*Ncolors + transposed

    # equals -1 for matrix inverse, +1 otherwise
    inverse = (1-2*inverted)

    # iterate over columns of B matrix
    for color in start:step:stop
		# the range of the checkerboard color
		n1 = colors[1,color]
		n2 = colors[2,color]
		I = neighbor_table[1,n1:n2]
		J = neighbor_table[2,n1:n2]
		C = coshΔτt[n1:n2]
		S = inverse * sinhΔτt[n1:n2]

        # perform multiply by checkerboard color
		M = sparse_color(length(v), I, J, C, S)
		checkerboard_color_lmul!(v, M)
    end

    return nothing
end


function checkerboard_lmul!(v::AbstractVector{T}, neighbor_table::Matrix{Int},
    coshΔτt::AbstractArray{E}, sinhΔτt::AbstractArray{E},
    colors::Matrix{Int}, L::Int; transposed::Bool=false, inverted::Bool=false) where {T<:Continuous, E<:Continuous}

    @assert !(T<:Real && E<:Complex) "Cannot multiply a real valued vector by a complex checkerboard matrix!"
    
    # number of checkerboard colors
    Ncolors = size(colors, 2)

    # how to iterate over neighbors in neighbor_table accounting for whether
    # or not the checkerboard matrix has been transposed
    transposed = inverted*(1-transposed) + (1-inverted)*transposed
    start      = (1-transposed) + transposed*Ncolors
    step       = 1 - 2*transposed
    stop       = (1-transposed)*Ncolors + transposed

    # iterate over columns of B matrix
    for color in start:step:stop

        # perform multiply by checkerboard color
		checkerboard_color_lmul!(v, colors[1,color],colors[2,color], neighbor_table, coshΔτt, sinhΔτt, L, inverted=inverted)
    end

    return nothing
end

function sparse_color(N::Int, I::AbstractVector{Int}, J::AbstractVector{Int},
					  C::AbstractVector{E}, S::AbstractVector{E}) where {E<:Continuous}

	@assert length(I) == length(J)
	@assert length(I) == length(S)
	@assert length(I) == length(C)

	# construct colptr, rowval, nzval directly
	colptr = Array{Int}(undef, N+1)
	rowval = Array{Int}(undef, 2*length(I)+N)
	nzval  = Array{E}(undef, 2*length(I)+N)

	iota = [Int(i) for i in range(1,N)]
	idxI = indexin(iota, I)
	idxJ = indexin(iota, J)
				   
	loc::Int = 1
	colptr[1] = loc
	for col in 1:N
		if (n = idxI[col]) != nothing
			rowval[loc]   = col
			rowval[loc+1] = J[n]
			nzval[loc]    = C[n]
			nzval[loc+1]  = conj(S[n])
			cur = 2
		elseif (n = idxJ[col]) != nothing
			rowval[loc]   = I[n]
			rowval[loc+1] = col
			nzval[loc]    = S[n]
			nzval[loc+1]  = C[n]
			cur = 2
		else
			# bystander index
			rowval[loc] = col
			nzval[loc] = 1.0
			cur = 1
		end
		loc += cur
		colptr[col+1] = loc
	end
	@assert loc == 2*length(I)+N+1

	return SparseArrays.SparseMatrixCSC{E,Int}(N,N,colptr,rowval,nzval)
end

@doc raw"""
    checkerboard_color_lmul!(v::AbstractVector{T}, start::Int, stop::Int, neighbor_table::Matrix{Int},
        coshΔτt::AbstractVector{E}, sinhΔτt::AbstractVector{E},
        transposed::Bool=false, inverted::Bool=false) where {T<:Continuous, E<:Continuous}

Multiply in-place the vector `v` by the `color` checkerboard color matrix.
"""
function checkerboard_color_lmul!(v::AbstractVector{T}, M::SparseArrays.SparseMatrixCSC{E,Int}
	) where {T<:Continuous, E<:Continuous}

    @assert !(T<:Real && E<:Complex) "Cannot multiply a real valued vector by a complex checkerboard matrix!"
	copy!(v, M*v)
	return nothing

    # iterate over neighbor pairs
	@fastmath @inbounds  for n in 1:length(I)
        # get pair of neighbor sites
        i = I[n]
        j = J[n]
        # get the relevant cosh and sinh values
        cᵢⱼ = coshΔτt[n]
        sᵢⱼ = sinhΔτt[n]
        # get the initial matrix elements
        vᵢ = v[i]
        vⱼ = v[j]
        # in-place multiply
        v[i] = cᵢⱼ * vᵢ + sᵢⱼ * vⱼ
        v[j] = cᵢⱼ * vⱼ + conj(sᵢⱼ) * vᵢ
    end

    return nothing
end


function checkerboard_color_lmul!(v::AbstractVector{T}, start::Int, stop::Int, neighbor_table::Matrix{Int},
    coshΔτt::AbstractVector{E}, sinhΔτt::AbstractVector{E}, L::Int; inverted::Bool=false) where {T<:Continuous, E<:Continuous}

    @assert !(T<:Real && E<:Complex) "Cannot multiply a real valued vector by a complex checkerboard matrix!"

    # equals -1 for matrix inverse, +1 otherwise
    inverse = (1-2*inverted)

    # iterate over neighbor pairs
    @fastmath @inbounds for n in start:stop
        # get pair of neighbor sites
        i = neighbor_table[1,n]
        j = neighbor_table[2,n]
        # get the relevant cosh and sinh values
        cᵢⱼ = coshΔτt[n]
        sᵢⱼ = inverse * sinhΔτt[n]
        # iterate over imaginary time slices
        @simd for τ in 1:L
            # get the indices
            k = (i-1)*L + τ
            l = (j-1)*L + τ
            # get the initial matrix elements
            vᵢ = v[k]
            vⱼ = v[l]
            # in-place multiply
            v[k] = cᵢⱼ * vᵢ + sᵢⱼ       * vⱼ
            v[l] = cᵢⱼ * vⱼ + conj(sᵢⱼ) * vᵢ
        end
    end

    return nothing
end


function checkerboard_color_lmul!(v::AbstractVector{T}, start::Int, stop::Int, neighbor_table::Matrix{Int},
    coshΔτt::AbstractMatrix{E}, sinhΔτt::AbstractMatrix{E},
    L::Int; inverted::Bool=false) where {T<:Continuous, E<:Continuous}

    @assert !(T<:Real && E<:Complex) "Cannot multiply a real valued vector by a complex checkerboard matrix!"

    # equals -1 for matrix inverse, +1 otherwise
    inverse = (1-2*inverted)

    # iterate over neighbor pairs
    @fastmath @inbounds for n in start:stop
        # get pair of neighbor sites
        i = neighbor_table[1,n]
        j = neighbor_table[2,n]
        # iterate over imaginary time slices
        @simd for τ in 1:L
            # get the relevant cosh and sinh values
            cᵢⱼ = coshΔτt[τ,n]
            sᵢⱼ = inverse * sinhΔτt[τ,n]
            # get the indices
            k = (i-1)*L + τ
            l = (j-1)*L + τ
            # get the initial matrix elements
            vᵢ = v[k]
            vⱼ = v[l]
            # in-place multiply
            v[k] = cᵢⱼ * vᵢ + sᵢⱼ       * vⱼ
            v[l] = cᵢⱼ * vⱼ + conj(sᵢⱼ) * vᵢ
        end
    end

    return nothing
end
