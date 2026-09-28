"""Small zero-copy solve with NVTX ranges for transfer inspection.

Run with the borrowed-vector PyAMGX extension and AMGX library. --copy-baseline
adds the old transfer path so a profiler can verify that it detects copies.
"""
import argparse
import json
from pathlib import Path

import cupy as cp
import numpy as np
import pyamgx
from cupyx.profiler import time_range


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dDDI", "dDFI", "dFFI"), default="dDDI")
    parser.add_argument("--copy-baseline", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    value_dtype = np.float32 if args.mode[2] == "F" else np.float64
    vector_dtype = np.float32 if args.mode[1] == "F" else np.float64
    objects = []

    def own(obj):
        objects.append(obj)
        return obj

    pyamgx.initialize()
    try:
        cfg = own(pyamgx.Config().create_from_dict({
            "config_version": 2, "exception_handling": 0,
            "solver": {"solver": "FGMRES", "preconditioner": {"solver": "NOSOLVER"},
                       "max_iters": 30, "gmres_n_restart": 8, "tolerance": 1e-6,
                       "monitor_residual": 1, "convergence": "RELATIVE_INI_CORE"}}))
        resources = own(pyamgx.Resources().create_simple(cfg))
        matrix = own(pyamgx.Matrix().create(resources, mode=args.mode))
        matrix.upload(np.array([0, 2, 4], np.int32), np.array([0, 1, 0, 1], np.int32),
                      np.array([4., 1., 1., 3.], value_dtype), shape=(2, 2))
        solver = own(pyamgx.Solver().create(resources, cfg, mode=args.mode))
        solver.setup(matrix)
        rhs_vector = own(pyamgx.Vector().create(resources, mode=args.mode))
        solution_vector = own(pyamgx.Vector().create(resources, mode=args.mode))
        producer = cp.cuda.Stream(non_blocking=True)
        with producer:
            rhs = cp.array([1., 2.], dtype=vector_dtype)
            solution = cp.zeros(2, dtype=vector_dtype)
            with time_range("borrowed.attach"):
                rhs_vector.attach(rhs)
                solution_vector.attach(solution)
            with time_range("borrowed.solve"):
                solver.solve(rhs_vector, solution_vector)
        # No manual AMGX-to-consumer synchronization or download is needed.
        with cp.cuda.Stream(non_blocking=True):
            residual = cp.array([[4., 1.], [1., 3.]], dtype=vector_dtype) @ solution - rhs
            residual_norm = float(cp.linalg.norm(residual).get())
        assert residual_norm < 2e-5
        assert rhs_vector.attached_ptr == rhs.data.ptr
        assert solution_vector.attached_ptr == solution.data.ptr
        report = {
            "mode": args.mode, "residual_norm": residual_norm,
            "rhs_ptr": rhs.data.ptr, "amgx_rhs_ptr": rhs_vector.attached_ptr,
            "solution_ptr": solution.data.ptr, "amgx_solution_ptr": solution_vector.attached_ptr,
            "cupy_version": cp.__version__, "pyamgx_extension": Path(pyamgx.__file__).name,
            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
            "cuda_driver": cp.cuda.runtime.driverGetVersion(),
            "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
            "amgx_libraries": sorted({Path(line.split()[-1]).name for line in
                Path("/proc/self/maps").read_text().splitlines() if "libamgx" in line}),
        }
        with time_range("borrowed.detach"):
            assert rhs_vector.detach() is rhs
            assert solution_vector.detach() is solution
        if args.copy_baseline:
            # A profiler control: exactly the transfers attach is replacing.
            with cp.cuda.Stream.null:
                with time_range("copy.upload"):
                    rhs_vector.upload(rhs)
                    solution_vector.upload(solution)
                with time_range("copy.download"):
                    solution_vector.download_raw(solution.data.ptr)
                cp.cuda.Stream.null.synchronize()
        rendered = json.dumps(report, indent=2)
        print(rendered)
        if args.report:
            args.report.write_text(rendered + "\n")
    finally:
        for obj in reversed(objects):
            obj.destroy()
        pyamgx.finalize()


if __name__ == "__main__":
    main()
