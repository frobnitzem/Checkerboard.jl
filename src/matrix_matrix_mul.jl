####################################
## MATRIX-MATRIX MULTIPLY METHODS ##
####################################

@doc raw"""
    checkerboard_lmul!(B::AbstractMatrix{T}, neighbor_table::AbstractMatrix{Int},
        coshΔτt::AbstractVector{T}, sinhΔτt::AbstractVector{T}, colors::AbstractMatrix{Int};
        transposed::Bool=false, inverted::Bool=false) where {T<:Continuous, E<:Continuous}

Evaluate the matrix-matrix product in-place `B=Γ⋅B` where `Γ` is the checkerboard matrix.
"""
function checkerboard_lmul!(B::AbstractMatrix{T}, neighbor_table::AbstractMatrix{Int},
    coshΔτt::AbstractVector{E}, sinhΔτt::AbstractVector{E},
    colors::AbstractMatrix{Int}; transposed::Bool=false, inverted::Bool=false) where {T<:Continuous, E<:Continuous}

    @assert !(T<:Real && E<:Complex) "Cannot multiply a real valued matrix by a complex checkerboard matrix!"

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

        # construct views for current checkerboard color
        n1,n2 = colors[1:2,color]
        nt = @view neighbor_table[:,n1:n2]
        ch = @view coshΔτt[n1:n2]
        sh = @view sinhΔτt[n1:n2]

        # perform multiply by checkerboard color
        checkerboard_color_lmul!(B, nt, ch, sh, inverted=inverted)
    end

    return nothing
end


@doc raw"""
    checkerboard_color_lmul!(B::AbstractMatrix{T}, neighbor_table::AbstractMatrix{Int},
        coshΔτt::AbstractVector{T}, sinhΔτt::AbstractVector{T},
        inverted::Bool=false) where {T<:Continuous, E<:Continuous}

Evaluate the matrix-matrix product in-place `B=Γ[c]⋅B` where `Γ[c]` is the `color` checkerboard color matrix.
"""
function checkerboard_color_lmul!(B::AbstractMatrix{T}, nt::AbstractMatrix{Int},
    ch::AbstractVector{E}, sh::AbstractVector{E};
    inverted::Bool=false) where {T<:Continuous, E<:Continuous}

    @assert !(T<:Real && E<:Complex) "Cannot multiply a real valued matrix by a complex checkerboard matrix!"

    # equals -1 for matrix inverse, +1 otherwise
    inverse = 1 - 2*inverted

    # iterate over columns of B
    @inbounds @fastmath for c in 1:size(B,2)
        # iterate over neighbor pairs
        @simd for n in 1:size(nt, 2)
            # get the pair of neighboring sites
            i = nt[1,n]
            j = nt[2,n]
            # get relevant cosh and sinh values
            cᵢⱼ = ch[n]
            sᵢⱼ = inverse * sh[n]
            # get relevant matrix elements
            bᵢ = B[i,c]
            bⱼ = B[j,c]
            # perform multiply
            B[i,c] = cᵢⱼ * bᵢ + sᵢⱼ * bⱼ
            B[j,c] = cᵢⱼ * bⱼ + conj(sᵢⱼ) * bᵢ
        end
    end

    return nothing
end


@doc raw"""
    checkerboard_rmul!(B::AbstractMatrix{T}, neighbor_table::AbstractMatrix{Int},
        coshΔτt::AbstractVector{E}, sinhΔτt::AbstractVector{E}, colors::AbstractMatrix{Int};
        transposed::Bool=false, inverted::Bool=false) where {T<:Continuous, E<:Continuous}

Evaluate the matrix-matrix product in-place `B=B⋅Γ` where `Γ` is the checkerboard matrix.
"""
function checkerboard_rmul!(B::AbstractMatrix{T}, neighbor_table::AbstractMatrix{Int},
    coshΔτt::AbstractVector{E}, sinhΔτt::AbstractVector{E},
    colors::AbstractMatrix{Int}; transposed::Bool=false, inverted::Bool=false) where {T<:Continuous, E<:Continuous}

    @assert !(T<:Real && E<:Complex) "Cannot multiply a real valued matrix by a complex checkerboard matrix!"

    # number of checkerboard colors
    Ncolors = size(colors, 2)

    # how to iterate over neighbors in neighbor_table accounting for whether
    # or not the checkerboard matrix has been transposed
    transposed = inverted*(1-transposed) + (1-inverted)*transposed
    start      = (1-transposed) + transposed*Ncolors
    step       = 1 - 2*transposed
    stop       = (1-transposed)*Ncolors + transposed

    # iterate over columns of B matrix
    for color in stop:-step:start

        # construct views for current checkerboard color
        n1,n2 = colors[1:2,color]
        nt = @view neighbor_table[:,n1:n2]
        ch = @view coshΔτt[n1:n2]
        sh = @view sinhΔτt[n1:n2]

        # perform multiply by checkerboard color
        checkerboard_color_rmul!(B, nt, ch, sh, inverted=inverted)
    end

    return nothing
end


@doc raw"""
    checkerboard_color_rmul!(B::AbstractMatrix{T}, neighbor_table::AbstractMatrix{Int},
        coshΔτt::AbstractVector{E}, sinhΔτt::AbstractVector{E};
        inverted::Bool=false) where {T<:Continuous, E<:Continuous}

Evaluate the matrix-matrix product in-place `B=B⋅Γ[c]` where `Γ[c]`
is the `color` checkerboard color matrix.
"""
function checkerboard_color_rmul!(B::AbstractMatrix{T}, nt::AbstractMatrix{Int},
    ch::AbstractVector{E}, sh::AbstractVector{E};
    inverted::Bool=false) where {T<:Continuous, E<:Continuous}

    @assert !(T<:Real && E<:Complex) "Cannot multiply a real valued matrix by a complex checkerboard matrix!"

    # equals -1 for matrix inverse, +1 otherwise
    inverse = 1 - 2*inverted

    # iterate over neighbor pairs
    @inbounds @fastmath for n in 1:size(nt, 2)
        # get the pair of neighboring sites
        j = nt[1,n]
        i = nt[2,n]
        # get relevant cosh and sinh values
        cᵢⱼ = ch[n]
        sᵢⱼ = inverse * sh[n]
        # iterate over rows of B matrix
        @simd for r in 1:size(B,1)
            # get relevant matrix elements
            bᵢ = B[r,i]
            bⱼ = B[r,j]
            # perform multiply
            B[r,i] = cᵢⱼ * bᵢ + sᵢⱼ * bⱼ
            B[r,j] = cᵢⱼ * bⱼ + conj(sᵢⱼ) * bᵢ
        end
    end

    return nothing
end
