import itertools
import numpy as np

def endpoint_simplex(q, b):
    best = np.zeros_like(b)
    cost = np.full(b.shape[:-1], np.inf)
    faces = [s for n in (1, 2, 3) for s in itertools.combinations(range(3), n)]
    for (left, right) in itertools.product(faces, repeat=2):
        ix = list(left) + [3 + i for i in right]
        part = q[..., ix, :][..., ix]
        target = b[..., ix]
        constraints = np.zeros((2, len(ix)))
        constraints[0, :len(left)] = 1.0
        constraints[1, len(left):] = 1.0
        rhs = np.concatenate([target[..., None], np.broadcast_to(constraints.T, target.shape + (2,))], axis=-1)
        solved = np.linalg.solve(part, rhs)
        u = solved[..., 0]
        normals = solved[..., 1:]
        lagrange = np.linalg.solve(np.einsum('ik,...kj->...ij', constraints, normals), np.einsum('ik,...k->...i', constraints, u) - 1.0)
        solution = u - np.einsum('...ki,...i->...k', normals, lagrange)
        feasible = (solution >= -1e-10).all(-1)
        solution = np.maximum(solution, 0.0)
        solution[..., :len(left)] /= solution[..., :len(left)].sum(-1, keepdims=True)
        solution[..., len(left):] /= solution[..., len(left):].sum(-1, keepdims=True)
        w = np.zeros_like(b)
        w[..., ix] = solution
        value = np.einsum('...i,...ij,...j->...', w, q, w) - 2 * (b * w).sum(-1)
        take = feasible & (value < cost)
        best = np.where(take[..., None], w, best)
        cost = np.where(take, value, cost)
    assert np.isfinite(cost).all() and (best >= 0).all()
    np.testing.assert_allclose(best[..., :3].sum(-1), 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(best[..., 3:].sum(-1), 1.0, rtol=0, atol=1e-12)
    return best

def simplex_quadratic(q, b):
    n = b.shape[-1]
    assert q.shape[-2:] == (n, n) and n in (3, 4)
    best = np.zeros_like(b)
    cost = np.full(b.shape[:-1], np.inf)
    for size in range(1, n + 1):
        for members in itertools.combinations(range(n), size):
            ix = list(members)
            part = q[..., ix, :][..., ix]
            target = b[..., ix]
            if size == 1:
                solution = np.ones_like(target)
            else:
                solved = np.linalg.solve(part, np.stack([target, np.ones_like(target)], axis=-1))
                u = solved[..., 0]
                normal = solved[..., 1]
                solution = u - normal * ((u.sum(-1) - 1) / normal.sum(-1))[..., None]
            feasible = (solution >= -1e-10).all(-1)
            solution = np.maximum(solution, 0)
            solution /= solution.sum(-1, keepdims=True)
            w = np.zeros_like(b)
            w[..., ix] = solution
            value = np.einsum('...i,...ij,...j->...', w, q, w) - 2 * (b * w).sum(-1)
            take = feasible & (value < cost)
            best = np.where(take[..., None], w, best)
            cost = np.where(take, value, cost)
    assert np.isfinite(cost).all() and np.min(best) >= 0
    np.testing.assert_allclose(best.sum(-1), 1, atol=1e-12)
    return best
