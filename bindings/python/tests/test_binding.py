import unittest
import sparsesolvers as ss
import numpy as np

class HomotopySolverTest(unittest.TestCase):
    def _test_smoke(self, N, T):
        solver = ss.Homotopy()

        A = np.identity(N, dtype=T)
        for n in range(0, N-1):
            signal = np.zeros(N, dtype=T)
            signal[n] = 1

            x, info = solver.solve(A, signal)
            np.array_equal(signal, x)

            assert(info.solution_error == 0)
            assert(info.iter == 1)

    def test_smoke_f32(self):
        self._test_smoke(5, np.float32)

    def test_smoke_f64(self):
        self._test_smoke(5, np.float64)

if __name__ == '__main__':
    print("[sparsesolvers] version={}".format(ss.version()))
    unittest.main()