import numpy as np
import quadprog

def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None):
    qp_G = .5 * (P + P.T)   # make sure P is symmetric
    qp_a = -q
    if A is not None:
        qp_C = -np.vstack([A, G]).T
        qp_b = -np.hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return quadprog.solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]


def point_polytope_shifted_hyperplane(p, vert):

    [dim1, N] = p.shape
    [dim2, M] = vert.shape
    if dim1 != dim2:
        print('Dimension error!')
        return

    # maximum margin classifier method
    ha = np.ones((dim1))
    hb = np.zeros((1))+0.00001
    hc = np.concatenate((ha,hb))
    H = np.diag(hc)
    f = np.zeros((dim1+1))
    Aa = np.concatenate((np.transpose(p), np.ones((N, 1))),axis=1)
    Ab = np.concatenate((-np.transpose(vert), -np.ones((M, 1))),axis=1)
    #Aa = np.transpose(p)
    #Ab = -np.transpose(vert)
    A = -np.transpose(np.concatenate((Aa,Ab),axis=0))
    b = np.ones((M + N))

    #sol = quadprog_solve_qp(H, f, A, b)
    sol = quadprog.solve_qp(H, f, A, b)[0]

    a = sol[0:dim1]
    b = -sol[-1]

    # normalization
    a_norm = np.linalg.norm(a)
    a = np.divide(a,a_norm)
    b = np.divide(b,a_norm)

    # shifted to be tight with the polytope
    min_dis = np.min(np.matmul(np.expand_dims(np.transpose(a),axis=0),vert) - b)
    b = b + min_dis

    return a, b

p = np.array([[-2],[1]])
vert = np.array([[-0.1160,1.6160,1.1160,-0.6160],[0.0670,1.0670,1.9330,0.9330]])

a,b = point_polytope_shifted_hyperplane(p,vert)

print(vert[0])