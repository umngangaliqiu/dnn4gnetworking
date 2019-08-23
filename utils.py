from typing import List

import numpy as np
import cvxpy as cp
import tensorflow as tf


def preprocess_data(loaddata, solardata, bus, alpha):
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
    #obj_dc = cp.Minimize(r_vector.T @ p_flow)
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


def cvx_ac(p, q, r_vector, x_vector, nm, branch, v_max, v_min):
    from_bus = branch[:, 0]
    to_bus = branch[:, 1]
    p_flow_child_sum = np.zeros(nm, dtype=object)
    q_flow_child_sum = np.zeros(nm, dtype=object)
    # left_con = []
    # right_con = []

    v_node = cp.Variable(nm)
    l_line = cp.Variable(nm)
    p_flow = cp.Variable(nm)
    q_flow = cp.Variable(nm)
    obj_ac = cp.Minimize(r_vector.T*l_line)
    for k in range(nm):
        p_flow_temp = 0
        q_flow_temp = 0
        for j in range(nm):
            if from_bus[j] == to_bus[k]:
                p_flow_temp = p_flow_temp + p_flow[to_bus[j]-2]
                q_flow_temp = q_flow_temp + q_flow[to_bus[j]-2]
            p_flow_child_sum[k] = p_flow_temp
            q_flow_child_sum[k] = q_flow_temp
        #
        # if from_bus[k] == 1:
        #     left_con.append(np.linalg.norm(np.hstack((2 * p_flow[int(to_bus[k]-2)], 2 * q_flow[int(to_bus[k]-2)], 1 - l_line[int(to_bus[k]-2)]))))
        #     right_con.append(l_line(int(to_bus[k]-2)) + 1)
        # else:
        #     left_con.append(np.linalg.norm(np.hstack((2 * p_flow[int(to_bus[k]-2)], 2 * q_flow[int(to_bus[k]-2)], v_node[int(from_bus[k]-2)]-l_line[int(to_bus[k]-2)]))))
        #     right_con.append(l_line(int(to_bus[k]-2)) + v_node[int(from_bus[k]-2)])

    constraints1 = [p_flow[int(to_bus[i]-2)] == p_flow_child_sum[i] + r_vector[int(to_bus[i]-2)]*l_line[int(to_bus[i]-2)] - p[int(to_bus[i]-2)] for i in range(nm)]
    constraints2 = [q_flow[int(to_bus[i]-2)] == q_flow_child_sum[i] + x_vector[int(to_bus[i]-2)]*l_line[int(to_bus[i]-2)] - q[int(to_bus[i]-2)] for i in range(nm)]
    constraints3 = [v_node[int(to_bus[i]-2)] == v_node[int(from_bus[i]-2)] + (np.power(r_vector[int(to_bus[i]-2)], 2)+np.power(x_vector[int(to_bus[i]-2)], 2))*l_line[int(to_bus[i]-2)] - 2*(r_vector[int(to_bus[i]-2)]*p_flow[int(to_bus[i]-2)] + x_vector[int(to_bus[i]-2)]*q_flow[int(to_bus[i]-2)]) for i in range(1, nm)]
    constraints31 = [v_node[int(to_bus[i]-2)] == 1 + (np.power(r_vector[int(to_bus[i]-2)], 2)+np.power(x_vector[int(to_bus[i]-2)], 2))*l_line[int(to_bus[i]-2)] - 2*(r_vector[int(to_bus[i]-2)]*p_flow[int(to_bus[i]-2)] + x_vector[int(to_bus[i]-2)]*q_flow[int(to_bus[i]-2)]) for i in range(1)]
    constraints4 = [cp.norm([2 * p_flow[int(to_bus[i]-2)]], 2) <= 10 for i in range(1, nm)]#l_line(int(to_bus[i]-2)) + v_node[int(from_bus[i]-2)] for i in range(1, nm)]

    constraints4 = [cp.norm([2 * p_flow[int(to_bus[i]-2)], 2 * q_flow[int(to_bus[i]-2)], v_node[int(from_bus[i]-2)]-l_line[int(to_bus[i]-2)]], 2) <= l_line(int(to_bus[i]-2)) + v_node[int(from_bus[i]-2)] for i in range(1, nm)]
    constraints41 = [cp.norm([2 * p_flow[int(to_bus[i]-2)], 2 * q_flow[int(to_bus[i]-2)], 1-l_line[int(to_bus[i]-2)]], 2) <= l_line[int(to_bus[i]-2)] + v_node[int(from_bus[i]-2)] for i in range(1)]


    #constraints4 = [left_con <= right_con]
    # constraints4 = [l_line[int(to_bus[i]-2)]*v_node[int(from_bus[i]-2)] >= np.power(p_flow[int(to_bus[i]-2)], 2) + np.power(q_flow[int(to_bus[i]-2)], 2) for i in range(1, nm)]
    # constraints41 = [l_line[int(to_bus[i]-2)]*1 >= np.power(p_flow[int(to_bus[i]-2)], 2) + np.power(q_flow[int(to_bus[i]-2)], 2) for i in range(1)]
    constraints5 = [v_node >= np.power(v_min, 2), v_node <= np.power(v_max, 2)]

    #prob = cp.Problem(obj_ac, constraints=constraints1+constraints2+constraints3+constraints31+constraints4+constraints41)
    prob = cp.Problem(obj_ac, constraints=constraints1 + constraints2 + constraints3 + constraints31 + constraints4 + constraints41)

    result_ac = prob.solve()

    return result_ac


def cvx_ac_matrix(p, q, r_vector, r_matrix, x_matrix, a_matrix, a_inv, a0, v0, bus, nm, v_max, v_min):
    ap = np.hstack((np.zeros((nm, nm)), np.dot(-a_inv.T, r_matrix)))
    bp = np.dot(a_inv, p)

    av1 = a_inv.dot(x_matrix)
    av2 = np.dot(-a_inv, (2 * np.dot(np.dot(r_matrix, a_inv.T), r_matrix) + np.dot(r_matrix, r_matrix) + np.dot(x_matrix, x_matrix)))
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
            an.append(np.vstack((2 * e_matrix[:, k].reshape(1, -1) @ ap,
                                np.hstack((2 * e_matrix[:, k].reshape(1, -1), np.zeros((1, nm)))),
                                np.hstack((np.zeros((1, nm)), -e_matrix[:, k].reshape(1, -1))))))
            # aaa=np.vstack((2 * e_matrix[:, k].reshape(1, -1) @ ap,
            #                     np.hstack((2 * e_matrix[:, k].reshape(1, -1), np.zeros((1, nm)))),
            #                     np.hstack((np.zeros((1, nm)), -e_matrix[:, k].reshape(1, -1)))))

            temp = np.zeros(3)
            temp[0] = np.dot(2 * e_matrix[:, k], bp)
            temp[2] = 1
            bn.append(temp)
            # print(np.shape(temp))

            cnn1 = np.zeros(2*nm)
            cnn1[:nm] = np.zeros(nm)
            cnn1[nm:] = e_matrix[k, :]

            cn.append(cnn1.T)
            # print(np.shape(cnn1.reshape(-1,1)))
            dn.append(np.ones(1))
        else:
            for j in range(nm):
                if a_matrix[k, j] == 1:
                    pik = j
                break

            an.append(np.vstack((2 * e_matrix[:, k].reshape(1, -1) @ ap,
                                 np.hstack((2 * e_matrix[:, k].reshape(1, -1), np.zeros((1, nm)))),
                                 e_matrix[:, pik].reshape(1, -1) @ av
                                 - np.hstack((np.zeros((1, nm)), e_matrix[:, k].reshape(1, -1))))))

            temp = np.zeros(3)
            temp[0] = np.dot(2 * e_matrix[k, :], bp)
            temp[2] = np.dot(e_matrix[pik, :], bv)

            bn.append(temp)
            temp = np.zeros(2*nm)
            temp[nm:] = e_matrix[k, :]
            cnn2 = np.dot(e_matrix[pik, :], av) + temp

            cn.append(cnn2)

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

def cvx_ac_matrix_simple(p, q, r_vector, r_matrix, x_matrix, a_matrix, a_inv, a0, v0, bus, nm, v_max, v_min):
    print(r_vector)
    z = cp.Variable(nm)
    r_z = np.zeros(2*nm)
    r_z[nm:] = r_vector
    obj_ac = cp.Minimize(r_vector.T*z)
    prob = cp.Problem(obj_ac)
    result_ac = prob.solve()
    print(z.value)

    return result_ac




