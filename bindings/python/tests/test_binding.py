'''
Python binding tests
'''

import unittest
import sparsesolvers as ss
import numpy as np

class HomotopySolverTest(unittest.TestCase):
    def _test_smoke(self, N, T):
        A = np.identity(N, dtype=T)
        solver = ss.Homotopy(A)

        for n in range(0, N-1):
            signal = np.zeros(N, dtype=T)
            signal[n] = 1

            x, info = solver.solve(signal)
            assert np.array_equal(signal, x)
            assert info.solution_error == 0
            assert info.iter == 1

    def test_smoke_f32(self):
        '''smoke test (float32)'''
        self._test_smoke(5, np.float32)

    def test_smoke_f64(self):
        '''smoke test (float64)'''
        self._test_smoke(5, np.float64)


    def test_row_subset(self):
        '''test a subset of rows'''

        A = np.random.rand(10, 5) * 0.1
        A_sub = A[:5, :] # subset of rows
        A_sub[:, 0] = 1  # needle to find

        signal = np.ones(5)
        x, info = ss.Homotopy(A_sub).solve(signal)

        assert len(x) == 5
        assert np.linalg.norm(x, 0) == 1

    def test_col_subset(self):
        '''test a subset of columns'''

        A = np.random.rand(10, 5) * 0.1
        A[:, 0] = 1 # column we'll be skipping
        A[:, 3] = 1 # needle to find

        A_sub = A[:, 2:]
        signal = np.ones(10)
        x, info = ss.Homotopy(A_sub).solve(signal)

        assert len(x) == 3
        assert np.argmax(x) == 1

    def test_transpose(self):
        '''test a transposed input'''

        A = np.random.rand(5, 10) * 0.1
        A[3, :] = 1 # needle to find as a row

        signal = np.ones(10)
        x, info = ss.Homotopy(A.T).solve(signal)

        assert len(x) == 5
        assert np.argmax(x) == 3


if __name__ == '__main__':
    print("[sparsesolvers] version={}".format(ss.version()))
    unittest.main()