#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.09.12'


import numpy as np


def forwardSubstitution(lowerTriangularMat, b):
    assert lowerTriangularMat.shape[1] == lowerTriangularMat.shape[0]
    assert lowerTriangularMat.shape[0] == b.shape[0]
    assert b.ndim == 2 and b.shape[1] == 1
    b_ = b.copy()
    x = np.zeros_like(b_)
    for col in range(lowerTriangularMat.shape[1]):
        if lowerTriangularMat[col][col] == 0:
            raise ValueError("lowerTriangularMat is singular")
        x[col] = b_[col] / lowerTriangularMat[col][col]
        for row in range(col + 1, b_.shape[0]):
            b_[row] -= lowerTriangularMat[row][col] * x[col]
    return x


def backwardSubstitution(upperTriangularMat, b):
    assert upperTriangularMat.shape[1] == upperTriangularMat.shape[0]
    assert upperTriangularMat.shape[0] == b.shape[0]
    assert b.ndim == 2 and b.shape[1] == 1
    b_ = b.copy()
    x = np.zeros_like(b_)
    for col in range(upperTriangularMat.shape[1] - 1, -1, -1):
        if upperTriangularMat[col][col] == 0:
            raise ValueError("upperTriangularMat is singular")
        x[col] = b_[col] / upperTriangularMat[col][col]
        for row in range(col - 1, -1, -1):
            b_[row] -= upperTriangularMat[row][col] * x[col]
    return x


def LU(mat):
    assert mat.shape[1] == mat.shape[0]
    L = np.eye(mat.shape[1])
    U = mat.copy()
    for col in range(U.shape[1]):
        if U[col][col] == 0:
            break
        for row in range(col + 1, U.shape[0]):
            L[row][col] = U[row][col] / U[col][col]
            U[row][col] = 0

        for i in range(col + 1, U.shape[0]):
            for j in range(col + 1, U.shape[1]):
                U[i][j] -= L[i][col] * U[col][j]
    return L, U


def LUPivoting(mat):
    assert mat.shape[1] == mat.shape[0]
    L = np.eye(mat.shape[1])
    U = mat.copy()
    P = np.eye(mat.shape[1])
    for col in range(U.shape[1]):
        max_idx = np.argmax(U[col:, col])
        if max_idx != 0:
            U[[col, col + max_idx]] = U[[col + max_idx, col]]  # swap rows
            P[[col, col + max_idx]] = P[[col + max_idx, col]]  # swap rows
        if U[col][col] == 0:
            raise ValueError("Pivot is zero")
        for row in range(col + 1, U.shape[0]):
            L[row][col] = U[row][col] / U[col][col]
            U[row][col] = 0

        for i in range(col + 1, U.shape[0]):
            for j in range(col + 1, U.shape[1]):
                U[i][j] -= L[i][col] * U[col][j]

    return P, P @ L, U


def exp2_3():
    import scipy.sparse
    import scipy.sparse.linalg
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import spsolve
    a = 2 ** 0.5 / 2
    A = csc_matrix([
        [0, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [a, 0, 0, -1, -a, 0, 0, 0, 0, 0, 0, 0, 0],
        [a, 0, 1, 0, a, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, -1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], # 6
        [0, 0, 0, 0, a, 1, 0, 0, -a, -1, 0, 0, 0],
        [0, 0, 0, 0, a, 0, 1, 0, a, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, -1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, a, 0, 0, -a, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, a, 0, 1, a, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, a, 1],
    ])
    b = csc_matrix([
        [0],
        [10],
        [0],
        [0],
        [0],
        [0], # 6
        [0],
        [15],
        [0],
        [20],
        [0],
        [0],
        [0],
    ])
    x = spsolve(A, b)
    print(x)
    print(A.dot(x))
    print(b.todense().reshape(-1))

    import numpy.testing as npTest
    f = [0] + x.tolist()
    npTest.assert_allclose(f[2], f[6])
    npTest.assert_allclose(f[3], 10)
    npTest.assert_allclose(a * f[1], f[4] + a*f[5])
    npTest.assert_allclose(a * f[1] + f[3] + a*f[5], 0, atol=1e-7)
    npTest.assert_allclose(f[4], f[8])
    npTest.assert_allclose(f[7], 0)
    npTest.assert_allclose(a * f[5] + f[6], a * f[9] + f[10])
    npTest.assert_allclose(a * f[5] + f[7] + a * f[9], 15)
    npTest.assert_allclose(f[10], f[13])
    npTest.assert_allclose(f[11], 20)
    npTest.assert_allclose(f[8] + a*f[9], a*f[12])
    npTest.assert_allclose(a*f[9] + f[11] + a*f[12], 0, atol=1e-7)
    npTest.assert_allclose(f[13] + a*f[12], 0)
    print(np.linalg.cond(A.todense(), 1))
    print(scipy.sparse.linalg.norm(A, 1) * scipy.sparse.linalg.norm(scipy.sparse.linalg.inv(A), 1))


def testSubstitution():
    A = np.array([
        [2, 4, -2],
        [0, 1, 1],
        [0, 0, 4]
    ]).astype(np.float)
    b = np.array([2, 4, 8]).reshape(3, 1).astype(np.float)
    x = backwardSubstitution(A, b)
    print(x)


def testLU():
    A = np.array([
        [2, 4, -2],
        [4, 9, -3],
        [-2, -3, 7]
    ]).astype(np.float)
    b = np.array([2, 8, 10]).reshape(3, 1).astype(np.float)
    L, U = LU(A)
    print(L)
    print(U)
    assert np.allclose(L.dot(U), A)

    # Ly = b, y = Ux
    y = forwardSubstitution(L, b)
    x = backwardSubstitution(U, y)
    print(x)
    # Ux = Mb
    M = np.linalg.inv(L)
    print(M)
    Mb = M.dot(b)
    print(Mb)
    x = backwardSubstitution(U, Mb)
    print(x)


def testLUPivoting():
    A = np.array([
        [1, 2, 2],
        [4, 4, 2],
        [4, 6, 4]
    ]).astype(np.float)
    b = np.array([2, 8, 10]).reshape(3, 1).astype(np.float)
    P, L, U = LUPivoting(A)
    print(L)
    print(U)
    assert np.allclose(L @ U, P @ A)


def assignment6():
    A = np.array([
        [1, -1, 0],
        [-1, 2, -1],
        [0, -1, 1]
    ]).astype(np.float)
    L, U = LU(A)
    print(L)
    print(U)
    # import scipy.linalg
    # P, L, U = scipy.linalg.lu(A)
    # print(P)
    # print(L)
    # print(U)
    print(L.dot(U))


def assignment5():
    A = np.array([
        [115, -45, -45],
        [-45, 115, -45],
        [-45, -45, 195]
    ]).astype(np.float)
    b = np.array([700, 1500, -2100]).reshape(3, 1).astype(np.float)
    L, U = LU(A)
    print(L)
    print(U)
    assert np.allclose(L.dot(U), A)
    # Ly = b, y = Ux
    y = forwardSubstitution(L, b)
    x = backwardSubstitution(U, y)
    print(x)


def main():
    # testSubstitution()
    # testLU()
    # testLUPivoting()
    # exp2_3()
    # assignment6()
    assignment5()


if __name__ == '__main__':
    main()
