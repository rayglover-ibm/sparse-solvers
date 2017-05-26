import unittest
import sparsesolvers as ss
import numpy as np

class HomotopySolverTest(unittest.TestCase):
    def test_smoke(self):
        solver = ss.Homotopy()

        N = 5
        A = np.identity(N)

        for n in range(0, N-1):
            signal = np.zeros(N)
            signal[n] = 1

            x, info = solver.solve(A, signal)
            np.array_equal(signal, x)

            assert(info.solution_error == 0)
            assert(info.iter == 1)

if __name__ == '__main__':
    print("[sparsesolvers] version={}".format(ss.version()))
    unittest.main()