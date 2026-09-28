"""A reusable solver that accepts CuPyX/CuPy objects directly."""
import cupy as cp
from cupyx.scipy.sparse import csr_matrix
import pyamgx


CONFIG = {
    "config_version": 2,
    "exception_handling": 0,
    "solver": {
        "solver": "FGMRES",
        "preconditioner": {"solver": "NOSOLVER"},
        "max_iters": 100,
        "gmres_n_restart": 20,
        "monitor_residual": 1,
        "convergence": "RELATIVE_INI_CORE",
        "tolerance": 1e-10,
    },
}


def main():
    csr = csr_matrix(cp.array([[4., 1.], [1., 3.]], dtype=cp.float64))
    rhs = cp.array([6., 7.], dtype=cp.float64)
    solution = cp.zeros(2, dtype=cp.float64)

    pyamgx.initialize()
    try:
        with pyamgx.ReusableSolver(CONFIG, mode="dDDI") as solver:
            solver.setup(csr)
            solver.solve(rhs, out=solution)
            cp.testing.assert_allclose(solution, [1., 2.], atol=1e-10)
            print(solution)  # [1. 2.], in the original CuPy array

            rhs *= 2
            solver.solve(rhs, out=solution)
            cp.testing.assert_allclose(solution, [2., 4.], atol=1e-10)
            print(solution)  # [2. 4.], same handles and setup
        # The with block called solver.destroy(); calling it again is harmless.
        solver.destroy()
    finally:
        pyamgx.finalize()


if __name__ == "__main__":
    main()
