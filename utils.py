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
    result_dc = prob.solve()

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
    ap = np.hstack((np.zeros((nm, nm)), -a_inv.T @r_matrix))
    bp = a_inv @ p

    av1 = a_inv @ x_matrix
    av2 = -a_inv @ (2 * r_matrix @ a_inv.T @ r_matrix + r_matrix @ r_matrix + x_matrix @ x_matrix)
    av = np.hstack((2*av1, av2))
    #print(np.shape(a0.reshape(-1,1)))
    bv = a_inv @ (2 * r_matrix @ a_inv.T @ p.reshape(-1,1) - a0.reshape(-1,1) @ v0.reshape(1,1))
    print(np.shape(bv))
    aq = np.hstack((a_matrix.T, x_matrix))
    zero_41 = np.zeros((1,nm))
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

            bn.append(np.vstack((2 * e_matrix[:, k].reshape(1, -1)@bp,0,1)))
            bnn=np.vstack((2 * e_matrix[:, k].reshape(1, -1) @ bp, 0, 1))
            print(np.shape(bnn))

            cnn1 = np.vstack((np.hstack((np.zeros((1, nm)), e_matrix[:, k].reshape(1, -1)))))
            cn.append(cnn1.reshape(-1,1))
            #cnn=np.vstack(())
            print(np.shape(cnn1.reshape(-1,1)))
            dn.append(np.ones((1,1)))
        else:
            for j in range(nm):
                if a_matrix[k, j] == 1:
                    pik = j
                break


            an.append(np.vstack((2 * e_matrix[:, k].reshape(1, -1) @ ap,
                           np.hstack((2 * e_matrix[:, k].reshape(1, -1), np.zeros((1,nm)))),e_matrix[:, pik].reshape(1, -1) @ av
                           - np.hstack((np.zeros((1,nm)), e_matrix[:, k].reshape(1, -1))))))

            bn.append(np.vstack((2 * e_matrix[:, k].reshape(1, -1)@bp,0,e_matrix[:, pik].reshape(1, -1)@bv)))
            cnn2 = np.vstack((e_matrix[:, pik].reshape(1, -1) @ av + np.hstack((np.zeros((1, nm)), e_matrix[:, k].reshape(1, -1)))))
            cn.append(cnn2.reshape(-1,1))
            dn.append(np.vstack((e_matrix[:, pik].reshape(1, -1)@bv)))
            dnn2=np.vstack((e_matrix[:, pik].reshape(1, -1)@bv))
            print('dnn', np.shape(dnn2))

    # cvx begin
    q_flow = cp.Variable((nm, 1))
    l_flow = cp.Variable((nm, 1))
    zz = cp.hstack((q_flow.T, l_flow.T)).T
    print(np.shape(zz))
    obj_ac = cp.Minimize(np.hstack((np.zeros((nm, 1)).reshape(1, -1), r_vector.T)) @ zz)

    constraints1 = [aq @ zz == q.reshape(-1,1)]
    constraints2 = [av @ zz + bv >= 2*v_min.reshape(-1,1)]
    constraints3 = [av @ zz + bv <= 1.2*v_max.reshape(-1,1)]
    soc_constraints = [cp.SOC(cn[i].T@ zz + dn[i], an[i] @ zz) for i in range(1,nm)]
    prob = cp.Problem(obj_ac,  soc_constraints+ constraints1+constraints2+constraints3)

    result_ac = prob.solve()
    return result_ac





