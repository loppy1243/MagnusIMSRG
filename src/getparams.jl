_getparam(sym) = getproperty(PARAMS, sym)
function _macro_getparams(syms; cnst)
    @assert all(x -> x isa Symbol, syms)

    ret = Expr(:block)
    for sym in syms
        expr = :($(esc(sym)) = _getparam($(Meta.quot(sym))))
        push!(ret.args, cnst ? Expr(:const, expr) : expr)
    end

    ret
end

macro getparams(syms::Symbol...)
    _macro_getparams(syms, cnst=true)
end
macro getparams(tuple_expr::Expr)
    @assert tuple_expr.head === :tuple

    _macro_getparams(tuple_expr.args, cnst=true)
end

macro localgetparams(syms::Symbol...)
    _macro_getparams(syms, cnst=false)
end
macro localgetparams(tuple_expr::Expr)
    @assert tuple_expr.head === :tuple

    _macro_getparams(tuple_expr.args, cnst=false)
end
