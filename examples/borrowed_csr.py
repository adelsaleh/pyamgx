"""Small zero-copy CuPyX CSR solve and NVTX transfer-profile fixture."""
import argparse
from contextlib import contextmanager
import json

import cupy as cp
from cupyx.scipy import sparse
from pathlib import Path
import pyamgx


@contextmanager
def region(name):
    cp.cuda.nvtx.RangePush(name)
    try:
        yield
    finally:
        cp.cuda.nvtx.RangePop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("dDDI", "dFFI"), default="dDDI")
    parser.add_argument("--report")
    parser.add_argument("--copy-baseline", action="store_true")
    args = parser.parse_args()
    dtype = cp.float64 if args.mode == "dDDI" else cp.float32
    pyamgx.initialize()
    objects = []
    def own(obj):
        objects.append(obj)
        return obj
    try:
        cfg = own(pyamgx.Config().create_from_dict({
            "config_version": 2, "exception_handling": 0,
            "solver": {"solver": "FGMRES", "preconditioner": {"solver": "NOSOLVER"},
                       "max_iters": 30, "gmres_n_restart": 8, "tolerance": 1e-6,
                       "monitor_residual": 1, "convergence": "RELATIVE_INI_CORE"}}))
        resources = own(pyamgx.Resources().create_simple(cfg))
        with cp.cuda.Stream(non_blocking=True):
            csr = sparse.csr_matrix(cp.asarray([[4, 1, 0], [1, 4, 1], [0, 1, 3]], dtype=dtype))
            expected = cp.asarray([1, 2, 3], dtype=dtype)
            rhs, result = csr @ expected, cp.zeros(3, dtype=dtype)
            A = own(pyamgx.Matrix().create(resources, args.mode))
            with region("csr.attach"):
                A.attach_CSR(csr)
            b = own(pyamgx.Vector().create(resources, args.mode)).attach(rhs)
            x = own(pyamgx.Vector().create(resources, args.mode)).attach(result)
            solver = own(pyamgx.Solver().create(resources, cfg, args.mode))
            with region("csr.setup"):
                solver.setup(A)
            before = A.attached_ptrs
            with region("csr.solve"):
                solver.solve(b, x)
            after = A.attached_ptrs
            producer = tuple(v.data.ptr for v in (csr.indptr, csr.indices, csr.data))
            assert before == after == producer
            residual = float(cp.linalg.norm(rhs - csr @ result))
            assert residual < 1e-4
            if args.copy_baseline:
                control = own(pyamgx.Matrix().create(resources, args.mode))
                with region("csr.copy_upload"):
                    control.upload_CSR(csr)
            solver.destroy()
            objects.remove(solver)
            with region("csr.detach"):
                assert A.detach() is csr
        report = dict(mode=args.mode, cupy_version=cp.__version__,
                      cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                      cuda_driver=cp.cuda.runtime.driverGetVersion(),
                      pyamgx_path=Path(pyamgx.__file__).name, producer_ptrs=producer,
                      native_before=before, native_after=after, residual=residual)
        print(json.dumps(report, indent=2))
        if args.report:
            with open(args.report, "w") as output:
                json.dump(report, output, indent=2)
    finally:
        for obj in reversed(objects):
            obj.destroy()
        pyamgx.finalize()


if __name__ == "__main__":
    main()
