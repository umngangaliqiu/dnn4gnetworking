import numpy as np
import cvxpy as cp
import scipy.io as sio

def pre_process_data(loaddata, solardata, bus, alpha):
    pc_max = bus[1:, 5]
    load_data_pc = np.dot(loaddata, np.diag(pc_max))
    qc_max = bus[1:, 7]
    load_data_qc = np.dot(loaddata, np.diag(qc_max))
    pg_max = bus[1:, 9]
    solardata = np.dot(solardata, np.diag(pg_max))

    pc = load_data_pc.T
    qc = load_data_qc.T
    pg = alpha*solardata
    pg = pg.T

    return pc, pg, qc


def cvx_dc(p, q, r_vector, r_matrix, x_matrix, a_matrix, a_inv, a0, v0, bus, nm, v_max, v_min):
    v_node = cp.Variable(nm)
    p_flow = cp.Variable(nm)
    q_flow = cp.Variable(nm)
    obj_dc = cp.Minimize(r_vector.T*(np.power(p_flow, 2) + np.power(q_flow, 2)))
    constraints = [a_matrix.T*p_flow == p,
                   a_matrix.T*q_flow == q,
                   v_node == 2 * a_inv * x_matrix * q_flow + 2 * a_inv * r_matrix * p_flow - a_inv.dot(a0) * v0
                   # v_node >= np.power(v_min, 2),
                   # v_node <= np.power(v_max, 2),
                   # q_flow[0] <= bus[0, 5],
                   # p_flow[0] >= bus[0, 6],
                   # q_flow[0] <= bus[0, 7],
                   # q_flow[0] >= bus[0, 8]
                   ]

    prob = cp.Problem(obj_dc, constraints)
    # prob = cp.Problem(obj_dc)
    result_dc = prob.solve()
    print(p_flow.value)

    return result_dc


def cvx_ac(p, q, r_vector, a_inv, a_matrix, r_matrix, x_matrix, v_min, v_max, nm, a0, v0):
    ap = np.hstack((np.zeros((nm, nm)), np.dot(-a_inv.T, r_matrix)))
    bp = np.dot(a_inv, p)

    av1 = a_inv.dot(x_matrix)
    av2 = np.dot(-a_inv, (2 * np.dot(np.dot(r_matrix, a_inv.T), r_matrix)
                          + np.dot(r_matrix, r_matrix) + np.dot(x_matrix, x_matrix)))
    av = np.hstack((2*av1, av2))
    # print(np.shape(a0.reshape(-1,1)))
    bv = np.dot(a_inv, 2 * np.dot(r_matrix, np.dot(a_inv.T, p)) - a0 * v0)
    # print(np.shape(bv))
    aq = np.hstack((a_matrix.T, x_matrix))
    an = []
    bn = []
    cn = []
    dn = []

    # define socp constraints w.s.p matrix-vector
    e_matrix = np.eye(nm)

    for k in range(nm):
        if k == 0:
            temp = np.zeros((3, nm*2))
            temp[0, :] = 2 * np.dot(e_matrix[k, :], ap)
            temp[1, :nm] = 2 * e_matrix[k, :]
            temp[2, nm:] = -e_matrix[k, :]
            an.append(temp)

            temp = np.zeros(3)
            temp[0] = np.dot(2 * e_matrix[:, k], bp)
            temp[2] = 1
            bn.append(temp)
            # print(np.shape(temp))

            temp = np.zeros(2*nm)
            temp[:nm] = np.zeros(nm)
            temp[nm:] = e_matrix[k, :]

            cn.append(temp.T)
            # print(np.shape(cnn1.reshape(-1,1)))
            dn.append(np.ones(1))
        else:
            for j in range(nm):
                if a_matrix[k, j] == 1:
                    pik = j
                break

            # an.append(np.vstack((2 * e_matrix[:, k].reshape(1, -1) @ ap,
            #                      np.hstack((2 * e_matrix[:, k].reshape(1, -1), np.zeros((1, nm)))),
            #                      e_matrix[:, pik].reshape(1, -1) @ av
            #                      - np.hstack((np.zeros((1, nm)), e_matrix[:, k].reshape(1, -1))))))
            temp = np.zeros((3, nm * 2))
            temp_2 = np.zeros(nm*2)
            temp_2[nm:] = e_matrix[k, :]
            temp[0, :] = 2 * np.dot(e_matrix[k, :], ap)
            temp[1, :nm] = 2 * e_matrix[k, :]
            temp[2, :] = np.dot(e_matrix[pik, :], av) - temp_2
            an.append(temp)

            temp = np.zeros(3)
            temp[0] = np.dot(2 * e_matrix[k, :], bp)
            temp[2] = np.dot(e_matrix[pik, :], bv)
            bn.append(temp)

            temp = np.zeros(2*nm)
            temp[nm:] = e_matrix[k, :]
            temp = np.dot(e_matrix[pik, :], av) + temp
            cn.append(temp)

            dn.append(np.dot(e_matrix[:, pik], bv))

    z = cp.Variable(2*nm)

    r_z = np.zeros(2*nm)
    r_z[nm:] = r_vector
    obj_ac = cp.Minimize(r_z.T@z)

    constraints_1 = [aq@z == q]
    constraints_2 = [av@z + bv >= 0.1*v_min]
    constraints_3 = [av@z + bv <= 10*v_max]

    soc_constraints = [cp.SOC(cn[i].T@z + dn[i], an[i]@z + bn[i]) for i in range(nm)]
    prob = cp.Problem(obj_ac, soc_constraints + constraints_1 + constraints_2 + constraints_3)

    result_ac = prob.solve()
    # print(z.value)
    return result_ac


def cvx_ac_qg(p, qc, r_vector, x_vector, a_inv, a_matrix, r_matrix, x_matrix, v_min, v_max, nm, a0, v0, branch, bus):
    # qc_temp = sio.loadmat("qc_old.mat")
    # qc = qc_temp['qc_old']
    # pc_temp = sio.loadmat("pc_old.mat")
    # pc = pc_temp['pc_old']
    # pg_temp = sio.loadmat("pg_old.mat")
    # pg = pg_temp['pg_old']
    # p = pg - pc


    from_bus = branch[:, 0]
    to_bus = branch[:, 1]

    p_flow_child_sum = np.zeros(nm, dtype=object)
    q_flow_child_sum = np.zeros(nm, dtype=object)

    constr = []
    constr1 = []
    left = []
    right = []

    qg = cp.Variable(nm)
    p_flow = cp.Variable(nm)
    q_flow = cp.Variable(nm)
    v_node = cp.Variable(nm)
    l_line = cp.Variable(nm)
    obj_ac = cp.Minimize(r_vector.T@l_line)

    for i in range(nm):
        p_flow_temp = 0
        q_flow_temp = 0
        for j in range(nm):
            if from_bus[j] == to_bus[i]:
                p_flow_temp = p_flow_temp + p_flow[to_bus[j] - 2]
                q_flow_temp = q_flow_temp + q_flow[to_bus[j] - 2]
            p_flow_child_sum[i] = p_flow_temp
            q_flow_child_sum[i] = q_flow_temp

        constr = constr + [p_flow[int(to_bus[i] - 2)] == p_flow_child_sum[i] + r_vector[int(to_bus[i] - 2)] *
                           l_line[int(to_bus[i] - 2)] - p[int(to_bus[i] - 2)]]
        constr = constr + [qg[int(to_bus[i] - 2)] + q_flow[int(to_bus[i] - 2)] == q_flow_child_sum[i] +
                           x_vector[int(to_bus[i] - 2)] * l_line[int(to_bus[i] - 2)] - qc[int(to_bus[i] - 2)]]

        if from_bus[i] == 1:
            constr = constr + [v_node[int(to_bus[i] - 2)] == 1 + (np.power(r_vector[int(to_bus[i] - 2)], 2) + np.power(x_vector[int(to_bus[i] - 2)], 2)) * l_line[
                             int(to_bus[i] - 2)] - 2 * (r_vector[int(to_bus[i] - 2)] * p_flow[int(to_bus[i] - 2)] + x_vector[int(to_bus[i] - 2)] * q_flow[int(to_bus[i] - 2)])]
            z1 = p_flow[int(to_bus[i] - 2)]
            z2 = q_flow[int(to_bus[i] - 2)]
            z3 = l_line[int(to_bus[i] - 2)]
            z4 = cp.hstack((2 * z1, 2 * z2, 1 - z3))
            z5 = cp.norm(z4, 2)
            left.append(z5)

            right.append(l_line[int(to_bus[i] - 2)] + 1)

        else:
            constr = constr + [v_node[int(to_bus[i] - 2)] == v_node[int(from_bus[i] - 2)] + (
                    np.power(r_vector[int(to_bus[i] - 2)], 2) + np.power(x_vector[int(to_bus[i] - 2)], 2)) * l_line[
                            int(to_bus[i] - 2)] - 2 * (
                                    r_vector[int(to_bus[i] - 2)] * p_flow[int(to_bus[i] - 2)] + x_vector[
                                int(to_bus[i] - 2)] * q_flow[int(to_bus[i] - 2)])]

            b1 = p_flow[int(to_bus[i] - 2)]
            b2 = q_flow[int(to_bus[i] - 2)]
            b3 = v_node[int(from_bus[i] - 2)] - l_line[int(to_bus[i] - 2)]
            b4 = cp.hstack((2*b1, 2*b2, b3))
            b5 = cp.norm(b4, 2)
            left.append(b5)
            right.append(l_line[int(to_bus[i] - 2)] + v_node[int(from_bus[i] - 2)])

    constr = constr + [v_node >= np.power(v_min, 2)]
    constr = constr + [v_node <= np.power(v_max, 2)]
    constr = constr + [left[k] <= right[k] for k in range(nm)]
    constr1 = [
        q_flow[0] <= bus[0, 5],
        p_flow[0] >= bus[0, 6],
        q_flow[0] <= bus[0, 7],
        q_flow[0] >= bus[0, 8],
        qg >= bus[1:, 12],
        qg <= bus[1:, 11]
        ]
    constraints = constr + constr1
    prob = cp.Problem(obj_ac, constraints)
    result_ac = prob.solve()
    print(result_ac)

    return result_ac


# def pre_process_cvx_ac_matrix(p, r_matrix, x_matrix, a_matrix, a_inv, a0, v0, nm):
#     ap = np.hstack((np.zeros((nm, nm)), np.dot(-a_inv.T, r_matrix)))
#     bp = np.dot(a_inv, p)
#
#     av1 = a_inv.dot(x_matrix)
#     av2 = np.dot(-a_inv, (
#                 2 * np.dot(np.dot(r_matrix, a_inv.T), r_matrix) + np.dot(r_matrix, r_matrix) + np.dot(x_matrix,
#                                                                                                       x_matrix)))
#     av = np.hstack((2 * av1, av2))
#     # print(np.shape(a0.reshape(-1,1)))
#     print(a_inv.shape)
#     bv = np.dot(a_inv, 2 * np.dot(r_matrix, np.dot(a_inv.T, p)) - a0 * v0)
#     # print(np.shape(bv))
#     aq = np.hstack((a_matrix.T, x_matrix))
#     an = []
#     bn = []
#     cn = []
#     dn = []
#
#     # define socp constraints w.s.p matrix-vector
#     e_matrix = np.eye(nm)
#
#     for k in range(nm):
#         if k == 0:
#             temp = np.zeros((3, nm * 2))
#             temp[0, :] = 2 * np.dot(e_matrix[k, :], ap)
#             temp[1, :nm] = 2 * e_matrix[k, :]
#             temp[2, nm:] = -e_matrix[k, :]
#             an.append(temp)
#
#             temp = np.zeros(3)
#             temp[0] = np.dot(2 * e_matrix[:, k], bp)
#             temp[2] = 1
#             bn.append(temp)
#             # print(np.shape(temp))
#
#             temp = np.zeros(2 * nm)
#             temp[:nm] = np.zeros(nm)
#             temp[nm:] = e_matrix[k, :]
#
#             cn.append(temp.T)
#             # print(np.shape(cnn1.reshape(-1,1)))
#             dn.append(np.ones(1))
#         else:
#             for j in range(nm):
#                 if a_matrix[k, j] == 1:
#                     pik = j
#                 break
#
#             temp = np.zeros((3, nm * 2))
#             temp[0, :] = 2 * np.dot(e_matrix[k, :], ap)
#             temp[1, :nm] = 2 * e_matrix[k, :]
#             temp[2, nm:] = np.dot(e_matrix[pik, :], av) - np.hstack((np.zeros((1, nm)), e_matrix[k, :]))
#             an.append(temp)
#
#             temp = np.zeros(3)
#             temp[0] = np.dot(2 * e_matrix[k, :], bp)
#             temp[2] = np.dot(e_matrix[pik, :], bv)
#             bn.append(temp)
#
#             temp = np.zeros(2 * nm)
#             temp[nm:] = e_matrix[k, :]
#             temp = np.dot(e_matrix[pik, :], av) + temp
#             cn.append(temp)
#
#             dn.append(np.dot(e_matrix[:, pik], bv))
#
#     return aq, av, bv, an, bn, cn, dn





