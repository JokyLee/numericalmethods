#!/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'Hao Li'
__date__ = '2020.09.20'


from functools import reduce
import numpy as np
from solvelinearlu import backwardSubstitution


def v2H(v):
    H = np.eye(v.shape[0]) - 2 * (v @ v.T) / (v.T @ v)
    return H


def applyVasH(v, u):
    return u - (2 * (v.T @ u) / (v.T @ v)) * v


def householderQR(A):
    R = A.copy()
    H = []
    for k in range(R.shape[1]):
        alpha_k = -np.sign(R[k, k]) * np.linalg.norm(R[k:, k])
        vk = np.zeros_like(R[:, k])
        vk[k:] = R[k:, k]
        ek = np.zeros_like(R[:, k])
        ek[k] = 1
        vk -= alpha_k * ek
        beta_k = vk.T @ vk
        H.append(v2H(vk.reshape(-1, 1)))
        if beta_k == 0:
            continue
        for j in range(k, R.shape[1]):
            gamma = vk.T @ R[:, j]
            R[:, j] -= (2 * gamma / beta_k) * vk
    Q = reduce(lambda a, b: a @ b, map(lambda h: h.T, H))
    return Q, R


def householderQRInPlace(A, b=None):
    for k in range(A.shape[1]):
        alpha_k = -np.sign(A[k, k]) * np.linalg.norm(A[k:, k])
        vk = np.zeros_like(A[:, k])
        vk[k:] = A[k:, k]
        ek = np.zeros_like(A[:, k])
        ek[k] = 1
        vk -= alpha_k * ek
        beta_k = vk.T @ vk
        if beta_k == 0:
            continue
        for j in range(k, A.shape[1]):
            gamma = vk.T @ A[:, j]
            A[:, j] -= (2 * gamma / beta_k)* vk
        if b is not None:
            gamma = vk.T @ b.ravel()
            b -= ((2 * gamma / beta_k)* vk).reshape(-1, 1)


def QRSolve(A, b):
    A_ = A.copy()
    b_ = b.copy()
    householderQRInPlace(A_, b_)
    x = backwardSubstitution(A_[:A_.shape[1]], b_[:A_.shape[1]])
    residual = np.linalg.norm(b_[A_.shape[1]:]) ** 2
    return x, residual


def main():
    A = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [-1, 1, 0],
        [-1, 0, 1],
        [0, -1, 1],
    ], dtype=np.float)
    b = np.array([1237, 1941, 2417, 711, 1177, 475], dtype=np.float).reshape(-1, 1)

    print(QRSolve(A, b))
    Q, R = np.linalg.qr(A)
    print(Q @ R)
    Q, R = householderQR(A)
    print(Q @ R)
    householderQRInPlace(A, b)
    print(A)
    print(b)
    # print(A)
    valid_shape = min(A.shape)
    x = backwardSubstitution(R[:valid_shape, :valid_shape], b[:valid_shape])
    print(x)


if __name__ == '__main__':
    main()
