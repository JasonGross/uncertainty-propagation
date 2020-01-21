import sympy as sym

__all__ = ["propagate_uncertainty"]

def propagate_uncertainty(expr, values=None, uncertainties=None, d=sym.symbols('d', cls=sym.Function)):
    """
    propagate_uncertainty takes a sympy expression, and optionally dicts mapping variables to their values and uncertainties, and returns the propagated uncertainty

    >>> x, y = sym.symbols('x y')
    >>> propagate_uncertainty(x * y)
    x**2*d(y)**2 + y**2*d(x)**2
    >>> propagate_uncertainty(x * y, {x:2, y:3})
    9*d(x)**2 + 4*d(y)**2
    >>> propagate_uncertainty(x * y, {x:20, y:30}, {x:2, y:3})
    7200
    """
    if values is None: values = {}
    if uncertainties is None: uncertainties = {}
    variables = list(expr.free_symbols)
    dv = (lambda v: sym.symbols('d(%s)' % v.name))
    # check that we aren't reusing variable names:
    for v in variables:
        if dv(v) in variables:
            raise ValueError('propagate_uncertainty cannot handle having both a variable named %s and a variable named %s' % (repr(str(v)), repr(str(dv(v)))))
    # calculate symbolic error
    sym_error = sum(sym.diff(expr, v)**2 * dv(v)**2 for v in variables)
    # substitute in first the uncertainties, then the values
    num_error = sym_error.subs(dict((dv(k), v) for k, v in uncertainties.items())).subs(values)
    # replace the remaining uncertainties with derivative functions, rather than custom-named variables
    # this allows the user to specify the form the differentials take
    #  without resulting in us substituting in arguments to the differential function
    #  (that is, we don't want to return 9*d(2)**2 + 4*d(3)**2 for propagate_uncertainty(x * y, {x:2, y:3})
    final_error = num_error.subs(dict((dv(k), d(k)) for k in variables))
    return final_error

if __name__ == "__main__":
    import doctest
    doctest.testmod()
