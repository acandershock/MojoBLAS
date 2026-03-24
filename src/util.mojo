# error checking blas_error_if(condition, __function_name__, "parameter_name", parameter)
fn blas_error_if[T: Stringable ](cond: Bool, caller: String, param: String, val: T) raises:
    if(cond) :
        raise Error(caller, " Error: ", param, " = ", String(val))
    
