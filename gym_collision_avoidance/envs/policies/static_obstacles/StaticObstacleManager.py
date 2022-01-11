import numpy as np
import quadprog

class StaticObstacleManager(object):
    def __init__(self):
        '''
        Class that can handle obstacles in the environment
        ego_agent = agent[0]
        obstacles = list of all obstacles with their 4 cornerpoints
        m = the amount of closest obstacles you want in the list. Default = 6
        '''
        # Get position of ego_agent in coordinates
        self.obstacle = []
        self.n_linear_constraints = 4
        #self.obstacles_in_range = []
        self.angular_map = None
        self.occupancy_grid = None

        # Environment limits
        self.env_constraints = []
        self.env_constraints.append(np.array([1, 0, 50]))
        self.env_constraints.append(np.array([0, 1, 50]))
        self.env_constraints.append(np.array([-1, 0, 50]))
        self.env_constraints.append(np.array([0, -1, 50]))

    def get_linear_constraints(self,agent,add_constraint=True):
        linear_constraints = []

        if add_constraint:

            for obs in self.obstacle:
                constraint , result = self.compute_constraint(np.expand_dims(agent.pos_global_frame,axis=1),np.transpose(obs),agent.radius)
                if result:
                    linear_constraints.append(constraint)

        # Environment limits
        i = 0
        while len(linear_constraints) +i <= self.n_linear_constraints:
            a = self.env_constraints[i][0:2]
            b = self.env_constraints[i][2]
            p = agent.pos_global_frame
            distance = np.abs(a[0]*p[0]+a[1]*p[1]+b)/np.linalg.norm(a)
            linear_constraints.append(np.concatenate((self.env_constraints[i], np.array([distance])),axis=0))
            i += 1

        linear_constraints_array = np.stack(linear_constraints,axis=0)
        # Order constraints by distance
        linear_constraints_ordered = linear_constraints_array[linear_constraints_array[:, 3].argsort()]

        return linear_constraints_ordered

    def compute_constraint(self,p, vert,radius):
        [dim1, N] = p.shape
        [dim2, M] = vert.shape
        if dim1 != dim2:
            print('Dimension error!')
            return

        # maximum margin classifier method
        ha = np.ones((dim1))
        hb = np.zeros((1)) + 0.00001
        hc = np.concatenate((ha, hb))
        H = np.diag(hc)
        f = np.zeros((dim1 + 1))
        Aa = np.concatenate((np.transpose(p), np.ones((N, 1))), axis=1)
        Ab = np.concatenate((-np.transpose(vert), -np.ones((M, 1))), axis=1)

        A = -np.transpose(np.concatenate((Aa, Ab), axis=0))
        b = np.ones((M + N))

        try:
            sol = quadprog.solve_qp(H, f, A, b)[0]

            a = sol[0:dim1]
            b = -sol[-1]

            # normalization
            a_norm = np.linalg.norm(a)
            a = np.divide(a, a_norm)
            b = np.divide(b, a_norm)

            # shifted to be tight with the polytope
            min_dis = np.min(np.matmul(np.expand_dims(np.transpose(a), axis=0), vert) - b)
            b = b + min_dis - radius*1.2

            # Compute minimum distance to constraint
            distance = np.abs(a[0]*p[0]+a[1]*p[1]+b)/np.linalg.norm(a)

            constraint = np.concatenate((a, np.array([b]),np.array([min_dis])),axis=0)

            return constraint, True
        except:
            print("No linear Constraint!!!")
            return [], False







