for op in (:+, :-, :*, :/, :\); @eval begin
    $op(as::Number...) = any(isnan, as) ? error("NaN detected!") : Base.$op(as...)
    $op(as...) = Base.$op(as...)
end; end
