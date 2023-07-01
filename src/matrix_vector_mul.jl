####################################
## MATRIX-VECTOR MULTIPLY METHODS ##
####################################

@doc raw"""
    checkerboard_lmul!(v::AbstractVector{T}, neighbor_table::AbstractMatrix{Int},
        coshΔτt::AbstractVector{E}, sinhΔτt::AbstractVector{E}, colors::AbstractMatrix{Int};
        transposed::Bool=false, inverted::Bool=false) where {T<:Continuous, E<:Continuous}

Multiply in-place the vector `v` by the checkerboard matrix.
"""
function checkerboard_lmul!(v::AbstractVector{T}, neighbor_table::AbstractMatrix{Int},
    coshΔτt::AbstractVector{E}, sinhΔτt::AbstractVector{E},
    colors::AbstractMatrix{Int}; transposed::Bool=false, inverted::Bool=false) where {T<:Continuous, E<:Continuous}

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
        # the range of the checkerboard color
        # colors[1:2, color]

        # construct views for current checkerboard color
        n1,n2 = colors[1:2,color]
        nt = @view neighbor_table[:,n1:n2]
        ch = @view coshΔτt[n1:n2]
        sh = @view sinhΔτt[n1:n2]

        # perform multiply by checkerboard color
        checkerboard_color_lmul!(v, nt,ch,sh, inverted=inverted)
    end

    return nothing
end


function checkerboard_lmul!(v::AbstractVector{T}, neighbor_table::AbstractMatrix{Int},
    coshΔτt::AbstractArray{E}, sinhΔτt::AbstractArray{E},
    colors::AbstractMatrix{Int}, L::Int; transposed::Bool=false, inverted::Bool=false) where {T<:Continuous, E<:Continuous}

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

        # construct views for current checkerboard color
        n1,n2 = colors[1:2,color]
        nt = @view neighbor_table[:,n1:n2]
        ch = @view coshΔτt[n1:n2]
        sh = @view sinhΔτt[n1:n2]

        # perform multiply by checkerboard color
        checkerboard_color_lmul!(v, nt,ch,sh, L, inverted=inverted)
    end

    return nothing
end


@doc raw"""
    checkerboard_color_lmul!(v::AbstractVector{T}, neighbor_table::AbstractMatrix{Int},
        coshΔτt::AbstractVector{E}, sinhΔτt::AbstractVector{E},
        inverted::Bool=false) where {T<:Continuous, E<:Continuous}

Multiply in-place the vector `v` by the color matrix.
"""
function checkerboard_color_lmul!(v::AbstractVector{T}, nt::AbstractMatrix{Int},
        ch::AbstractVector{E}, sh::AbstractVector{E};
        inverted::Bool=false) where {T<:Continuous, E<:Continuous}

    @assert !(T<:Real && E<:Complex) "Cannot multiply a real valued vector by a complex checkerboard matrix!"

    # equals -1 for matrix inverse, +1 otherwise
    inverse = (1-2*inverted)

    # iterate over neighbor pairs
    @fastmath @inbounds for n in 1:size(nt, 2)
        # get pair of neighbor sites
        i = nt[1,n]
        j = nt[2,n]
        # get the relevant cosh and sinh values
        cᵢⱼ = ch[n]
        sᵢⱼ = inverse * sh[n]
        # get the initial matrix elements
        vᵢ = v[i]
        vⱼ = v[j]
        # in-place multiply
        v[i] = cᵢⱼ * vᵢ + sᵢⱼ * vⱼ
        v[j] = cᵢⱼ * vⱼ + conj(sᵢⱼ) * vᵢ
    end

    return nothing
end


function checkerboard_color_lmul!(v::AbstractVector{T}, nt::AbstractMatrix{Int},
    ch::AbstractVector{E}, sh::AbstractVector{E}, L::Int;
    inverted::Bool=false) where {T<:Continuous, E<:Continuous}

    @assert !(T<:Real && E<:Complex) "Cannot multiply a real valued vector by a complex checkerboard matrix!"

    # equals -1 for matrix inverse, +1 otherwise
    inverse = (1-2*inverted)

    # iterate over neighbor pairs
    @fastmath @inbounds for n in 1:size(nt, 2)
        # get pair of neighbor sites
        i = nt[1,n]
        j = nt[2,n]
        # get the relevant cosh and sinh values
        cᵢⱼ = ch[n]
        sᵢⱼ = inverse * sh[n]
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


function checkerboard_color_lmul!(v::AbstractVector{T}, nt::AbstractMatrix{Int},
    ch::AbstractMatrix{E}, sh::AbstractMatrix{E},
    L::Int; inverted::Bool=false) where {T<:Continuous, E<:Continuous}

    @assert !(T<:Real && E<:Complex) "Cannot multiply a real valued vector by a complex checkerboard matrix!"

    # equals -1 for matrix inverse, +1 otherwise
    inverse = (1-2*inverted)

    # iterate over neighbor pairs
    @fastmath @inbounds for n in 1:size(nt, 2)
        # get pair of neighbor sites
        i = nt[1,n]
        j = nt[2,n]
        # iterate over imaginary time slices
        @simd for τ in 1:L
            # get the relevant cosh and sinh values
            cᵢⱼ = ch[τ,n]
            sᵢⱼ = inverse * sh[τ,n]
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
