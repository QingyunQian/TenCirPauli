using JSON
using PauliPropagation
using Printf
using Statistics
using LinearAlgebra

function parse_args()
    values = Dict{String, String}(
        "mode" => "benchmark",
        "nqubits" => "28",
        "layers" => "6",
        "max_weight" => "5",
        "repeats" => "5",
        "iterations" => "20",
        "learning_rate" => "0.05",
    )
    index = 1
    while index <= length(ARGS)
        argument = ARGS[index]
        startswith(argument, "--") || error("unexpected argument: $argument")
        key = replace(argument[3:end], '-' => '_')
        index += 1
        index <= length(ARGS) || error("missing value for --$key")
        values[key] = ARGS[index]
        index += 1
    end
    return values
end

function parameters(nqubits, layers)
    return collect(range(-0.17, stop=0.23, length=nqubits * layers))
end

function brickwork_pairs(nqubits, layer)
    start = iseven(layer) ? 1 : 2
    return [(wire, wire + 1) for wire in start:2:(nqubits - 1)]
end

function observable(nqubits)
    terms = PauliString[]
    for wire in 1:nqubits
        push!(terms, PauliString(nqubits, :X, wire, -0.5))
    end
    for wire in 1:(nqubits - 1)
        push!(terms, PauliString(nqubits, [:Z, :Z], [wire, wire + 1], -1.0))
    end
    return VectorPauliSum(terms)
end

function circuit(nqubits, layers)
    gates = Gate[]
    for layer in 0:(layers - 1)
        for wire in 1:nqubits
            push!(gates, PauliRotation(:Y, wire))
        end
        for (wire0, wire1) in brickwork_pairs(nqubits, layer)
            push!(gates, CliffordGate(:CNOT, [wire0, wire1]))
        end
    end
    return gates
end

function value_and_gradient(gates, psum, params, max_weight)
    return rewindgradient(
        gates,
        psum,
        params,
        overlapwithzero;
        max_weight=max_weight,
        min_abs_coeff=0.0,
    )
end

function truncated_expectation(gates, psum, params, max_weight)
    propagated = propagate(gates, psum, params; max_weight=max_weight, min_abs_coeff=0.0)
    return overlapwithzero(propagated)
end

function finite_difference(gates, psum, params, max_weight; step=1.0e-6)
    gradient = zeros(Float64, length(params))
    for index in eachindex(params)
        plus = copy(params)
        minus = copy(params)
        plus[index] += step
        minus[index] -= step
        gradient[index] = (
            truncated_expectation(gates, psum, plus, max_weight)
            - truncated_expectation(gates, psum, minus, max_weight)
        ) / (2 * step)
    end
    return gradient
end

function run_correctness()
    nqubits, layers, max_weight = 8, 2, 8
    params = parameters(nqubits, layers)
    gates = circuit(nqubits, layers)
    psum = observable(nqubits)
    value, gradient = value_and_gradient(gates, psum, params, max_weight)
    truncated_max_weight = 5
    truncated_value, truncated_gradient = value_and_gradient(gates, psum, params, truncated_max_weight)
    truncated_fd_gradient = finite_difference(gates, psum, params, truncated_max_weight)
    return Dict(
        "nqubits" => nqubits,
        "layers" => layers,
        "max_weight" => max_weight,
        "nparameters" => length(params),
        "julia_value" => value,
        "julia_gradient" => gradient,
        "truncated_max_weight" => truncated_max_weight,
        "truncated_value" => truncated_value,
        "truncated_gradient" => truncated_gradient,
        "truncated_fd_gradient" => truncated_fd_gradient,
        "truncated_fd_gradient_max_abs_error" => maximum(abs.(truncated_gradient - truncated_fd_gradient)),
        "julia_threads" => Threads.nthreads(),
    )
end

function run_benchmark(nqubits, layers, max_weight, repeats, iterations, learning_rate)
    params = parameters(nqubits, layers)
    setup_start = time_ns()
    gates = circuit(nqubits, layers)
    psum = observable(nqubits)
    setup_seconds = (time_ns() - setup_start) / 1e9

    first_start = time_ns()
    first_value, first_gradient = value_and_gradient(gates, psum, params, max_weight)
    first_seconds = (time_ns() - first_start) / 1e9

    steady_seconds = Float64[]
    value = first_value
    gradient = first_gradient
    for _ in 1:repeats
        start = time_ns()
        value, gradient = value_and_gradient(gates, psum, params, max_weight)
        push!(steady_seconds, (time_ns() - start) / 1e9)
    end

    vqe_params = copy(params)
    vqe_start = time_ns()
    for _ in 1:iterations
        value, gradient = value_and_gradient(gates, psum, vqe_params, max_weight)
        vqe_params .-= learning_rate .* gradient
    end
    vqe_seconds = (time_ns() - vqe_start) / 1e9

    return Dict(
        "implementation" => "PauliPropagation.jl",
        "nqubits" => nqubits,
        "layers" => layers,
        "max_weight" => max_weight,
        "nparameters" => length(params),
        "first_value_and_grad_seconds" => first_seconds,
        "steady_value_and_grad_seconds_median" => median(steady_seconds),
        "steady_value" => value,
        "first_value" => first_value,
        "first_gradient" => first_gradient,
        "first_gradient_l2" => norm(first_gradient),
        "vqe_iterations" => iterations,
        "vqe_seconds" => vqe_seconds,
        "vqe_final_value" => value,
        "setup_seconds" => setup_seconds,
        "julia_threads" => Threads.nthreads(),
    )
end

function main()
    args = parse_args()
    mode = args["mode"]
    result = if mode == "correctness"
        run_correctness()
    elseif mode == "benchmark"
        run_benchmark(
            parse(Int, args["nqubits"]),
            parse(Int, args["layers"]),
            parse(Int, args["max_weight"]),
            parse(Int, args["repeats"]),
            parse(Int, args["iterations"]),
            parse(Float64, args["learning_rate"]),
        )
    else
        error("mode must be correctness or benchmark")
    end

    println("RESULT_JSON=" * JSON.json(result))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
