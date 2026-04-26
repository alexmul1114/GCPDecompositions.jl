
using ForwardDiff
using GCPDecompositions
using BenchmarkTools

function manual_deriv_time(loss_func, xvals, mvals)
    s = 0.0
    for x in xvals, m in mvals
        s += GCPLosses.deriv(loss_func, x, m)
    end
    return s
end

function autodiff_time(loss_func, xvals, mvals)
    s = 0.0
    for x in xvals, m in mvals
        s += ForwardDiff.derivative(m -> GCPLosses.value(loss_func, x, m), m)
    end
    return s
end

for (loss, (xvals, mvals)) in [
    GCPLosses.LeastSquares() => (-2:0.001:2, -2:0.001:2),
    GCPLosses.NonnegativeLeastSquares() => (0:0.001:2, 0:0.001:2),
    GCPLosses.Poisson() => (0:0.001:3, 0:0.001:3),
    GCPLosses.PoissonLog() => (-2:0.001:2, -2:0.001:2),
    GCPLosses.Gamma() => (0:0.001:2, 0.0:0.001:2),
    GCPLosses.Rayleigh() => (0:0.001:2, 0:0.001:2),
    GCPLosses.BernoulliOdds() => (0:0.001:2, 0:0.001:2),
    GCPLosses.BernoulliLogit() => (-2:0.001:2, -2:0.001:2),
    GCPLosses.NegativeBinomialOdds(1) => (0:0.001:2, 0:0.001:2),
    GCPLosses.Huber(1) => (-2:0.001:2, -2:0.001:2),
    GCPLosses.BetaDivergence(0) => (0:0.001:3, 0.1:0.001:3),
    GCPLosses.BetaDivergence(0.5) => (0:0.001:3, 0.1:0.001:3),
    GCPLosses.BetaDivergence(1) => (0:0.001:3, 0.1:0.001:3),
]
    @info "Loss=$loss:"
    @info "Manual derivative:"
    @btime manual_deriv_time($loss, $xvals, $mvals)
    @info "ForwardDiff:"
    @btime autodiff_time($loss, $xvals, $mvals)
end


