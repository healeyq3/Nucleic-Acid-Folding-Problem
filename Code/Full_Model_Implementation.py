from gurobipy import GRB, Model, quicksum
import numpy as np
import plotly.express as px
import itertools
import pandas as pd

class FoldingModel():

    def __init__(self, nucleotide_sequence, name, min_dist):
        self.ns = nucleotide_sequence.lower()
        self.n = len(nucleotide_sequence)
        self.nsa = np.array([char for char in nucleotide_sequence.lower()])
        self.min_dist = min_dist

        # Array used for indexing
        self.possible_pairs = [(i, j) for i in range(0, self.n-1) for j in range(i+1, self.n)]
        
        # Build character arrays
        self.a, self.u, self.c, self.g = self.build_char_arrays()

        # Initialize Gurobi Model
        name = "NAFM_t_%s_l_%s" % (name, self.n)
        self.model = Model(name)
        self.model.setParam('TimeLimit', 5*60)
        self.model.setParam('LogToConsole', 0)

        # Initialize Decision Variables
        self.x = self.model.addVars(self.possible_pairs, name = "x", vtype = GRB.BINARY)
    
    def build_char_arrays(self):
        a = np.where(self.nsa == "a", 1, 0)
        u = np.where(self.nsa == "u", 1, 0)
        c = np.where(self.nsa == "c", 1, 0)
        g = np.where(self.nsa == "g", 1, 0)
        return a, u, c, g

    def build_model(self, num_crossings):
        self.declare_objective()
        self.construct_constraints(num_crossings)

    def solve_model(self):
        self.model.optimize()

    def declare_objective(self):
        self.model.setObjective(quicksum([self.x[i, j] for (i, j) in self.possible_pairs]),
                                GRB.MAXIMIZE)
    
    def construct_constraints(self, num_crosses):                    
        self.single_match_constraints()
        self.complementary_constraints()
        self.min_dist_constraint()
        self.crossing_constraint(num_crosses)
        
    
    def single_match_constraints(self):
        self.model.addConstrs( (( quicksum([self.x[i, k] for i in range(0, k)]) 
                                + quicksum([self.x[k, j] for j in range(k + 1, self.n)]) )
                                <= 1 for k in range(0, self.n)), name = "sing_match_constraints"
                                )  
    
    def complementary_constraints(self):
        non_complementary_pairs = [(i, j) for (i, j) in self.possible_pairs
                                    if (self.a[i] + self.a[j] + self.c[i] + self.c[j] == 2)
                                    or (self.a[i] + self.a[j] + self.g[i] + self.g[j] == 2)
                                    or (self.u[i] + self.u[j] + self.c[i] + self.c[j] == 2)
                                    or (self.u[i] + self.u[j] + self.g[i] + self.g[j] == 2)]
        
        self.model.addConstrs((self.x[i, j] == 0 for (i, j) in non_complementary_pairs),
                                name = "allowed_bases")
    
    def min_dist_constraint(self):
        too_close_pairs = [(i, j) for (i, j) in self.possible_pairs if min(j - i, self.n - j + i) < self.min_dist]
        self.model.addConstrs((self.x[i, j] == 0 for (i, j) in too_close_pairs),
                                name = "min_dist_pairs")
    
    def crossing_constraint(self, num_crosses):
        # Create crossing decision variables
        indices = [(i, j, k, l) for (i, j) in self.possible_pairs for (k, l) in self.possible_pairs
                    if i < k < j < l]
        C = self.model.addVars(indices, name = "C", vtype = GRB.BINARY)

        # Main Constraint
        self.model.addConstr( ( quicksum(C[i, j, k, l] for (i, j, k, l) in indices ) <= num_crosses ),
                                name = "crossing_constraint_main" )

        # Supporting Constraints
        self.model.addConstrs( (self.x[i, j] + self.x[k, l] >= 2 * C[i, j, k, l] for (i, j, k, l) in indices),
                                name = "crossing_constraint_1")
        self.model.addConstrs( (self.x[i, j] + self.x[k, l] - 1 <= C[i, j, k, l] for (i, j, k, l) in indices),
                                name = "crossing_constraint_2" )


    def model_size(self):
        return self.model.NumVars + self.model.NumConstrs


class CrudeModel(FoldingModel):

    def __init__(self, nuc_sequence, name):
        super().__init__(nuc_sequence, name, 1)

class SimpleEnhancedModel(FoldingModel):

    def __init__(self, nuc_sequence, name):
        # Notice that we change the minimum distance between two pairs with the super call
        super().__init__(nuc_sequence, name, 3)

    # Override the old objective
    def declare_objective(self):
        weight_dict = {
            ("a", "u") : 2,
            ("u", "a") : 2,
            ("g", "c") : 3,
            ("c", "g") : 3,
            ("a", "c") : 0.05,
            ("c", "a") : 0.05,
            ("g", "u") : 0.1,
            ("u", "g") : 0.1
        }
        f = {(i, j) : (weight_dict[self.ns[i], self.ns[j]] if (self.ns[i], self.ns[j]) in weight_dict.keys() else 0)
                        for (i, j) in self.possible_pairs}
        self.model.setObjective(quicksum([f[i, j] * self.x[i, j] for (i, j) in self.possible_pairs]),
                                GRB.MAXIMIZE)
    
    # Override the old construct_constraints to no longer restrict base pairs
    # as the objective function itself now handles this
    def construct_constraints(self, num_crosses):
        self.single_match_constraints()
        self.min_dist_constraint()
        self.crossing_constraint(num_crosses)

class QuartetModel(FoldingModel):

    def __init__(self, nuc_sequence, name):
        super().__init__(nuc_sequence, name, 1)
        # Extra Indices that will be used
        self.ij_indices = [(i, j) for i in range(0, self.n - 3) for j in range(i+3, self.n)]
        self.ij_indicesP = [(i, j) for i in range(1, self.n - 3) for j in range(i + 3, self.n - 1)]
        self.nj_j_indices = [j for j in range(2, self.n - 1)]
        self.nj_indices = [(self.n - 1, j) for j in self.nj_j_indices]
        self.quartet_indicesP = self.ij_indicesP + self.nj_indices
        self.quartet_indices = self.ij_indices + self.nj_indices
        self.Q = self.model.addVars([(i, j) for i in range(self.n) for j in range(self.n)], name = "Q", vtype = GRB.BINARY)
        self.P = self.model.addVars([(i, j) for i in range(self.n) for j in range(self.n)], name = "P", vtype = GRB.BINARY)
        self.L = self.model.addVars([(i, j) for i in range(self.n) for j in range(self.n)], name = "L", vtype = GRB.BINARY)

    def construct_constraints(self, num_crosses):
        super().construct_constraints(num_crosses)
        self.add_quartet_constraints()

    def declare_objective(self):
        nj_j_indices = self.nj_j_indices
        quartet_indices = self.quartet_indices
        Q = self.Q
        P = self.P
        L = self.L
        n = self.n
        ns = self.ns
        weight_dict = {
            ('a', 'u', 'a', 'u') : 9,
            ('a', 'u', 'c', 'g') : 21,
            ('a', 'u', 'g', 'c') : 24,
            ('a', 'u', 'u', 'a') : 13,
            ('c', 'g', 'a', 'u') : 22,
            ('c', 'g', 'c', 'g') : 33,
            ('c', 'g', 'g', 'c') : 34,
            ('c', 'g', 'u', 'a') : 24,
            ('g', 'c', 'a', 'u') : 21,
            ('g', 'c', 'c', 'g') : 24,
            ('g', 'c', 'g', 'c') : 33,
            ('g', 'c', 'u', 'a') : 21,
            ('u', 'a', 'a', 'u') : 12,
            ('u', 'a', 'c', 'g') : 23,
            ('u', 'a', 'g', 'c') : 21,
            ('u', 'a', 'u', 'a') : 16,
        }
        w = {(i, j) : (weight_dict[ns[i], ns[j], ns[i + 1], ns[j - 1]]
                        if (ns[i], ns[j], ns[i + 1], ns[j - 1]) in weight_dict.keys() else 1)
                for (i, j) in self.ij_indices}
        wn = {(n - 1, j) : (weight_dict[ns[n - 1], ns[j], ns[0], ns[j - 1]]
                            if (ns[n - 1], ns[j], ns[0], ns[j - 1]) in weight_dict.keys() else 1)
                for j in nj_j_indices}
        w.update(wn)
        self.model.setObjective(
            (quicksum([self.x[i, j] for (i, j) in self.possible_pairs]) )
            + (quicksum([w[i, j]*(Q[i, j] + P[i, j] + L[i, j]) for (i, j) in quartet_indices])),
            GRB.MAXIMIZE
        )

    
    def add_quartet_constraints(self):
        n = self.n
        ij_indices = self.ij_indices
        ij_indicesP = self.ij_indicesP
        nj_j_indices = self.nj_j_indices
        nj_indices = self.nj_indices
        Q = self.Q
        P = self.P
        L = self.L

        # Basic Quartet Constraints
        self.model.addConstrs( (2 * Q[i, j] <= self.x[i, j] + self.x[i + 1, j - 1] for (i, j) in ij_indices),
                                name = "Quartet_Constr." )
        self.model.addConstrs( (2*Q[self.n - 1, j] <= self.x[j, self.n-1] + self.x[0, j - 1] for j in nj_j_indices),
                                    name = "Quartet_Constr." )
        self.model.addConstrs( (Q[i, j] == 0 for i, j in list(set(itertools.product(range(self.n), range(self.n))) - set().union(ij_indices, nj_indices))), name = "Quartet_Constr." )

        # Position of Quartet
        self.model.addConstrs((Q[i, j] >= P[i, j] for (i, j) in ij_indices), name = "Quartet_Position")
        self.model.addConstrs((Q[i, j] + (1 - Q[i - 1, j + 1]) >= 2*P[i, j] for (i, j) in ij_indicesP),
                                name = "Quartet_Position")
        self.model.addConstrs((Q[i, n - 1] + (1 - Q[0, i - 1]) >= 2*P[i, n - 1] for i in range(1, n-3)),
                                name = "Quartet_Position") # changed this one because Q wasn't defined for their range
        self.model.addConstrs((Q[0, j] + (1 - Q[n - 1, j + 1]) >= 2 * P[0, j] for j in range(3, n - 2)),
                                name = "Quartet_Position")
        self.model.addConstrs((Q[n - 1, j] >= P[n - 1, j] for j in range(2, n - 1)),
                                name = "Quartet_Position")
        self.model.addConstrs((Q[n - 1, j] + (1 - Q[n - 2, j + 1]) >= 2 * P[n - 1, j] for j in range(2, n - 1)),
                                name = "Quartet_Position") # changed to - 5 instead of - 3
        
        self.model.addConstrs((Q[i, j] >= L[i, j] for (i, j) in ij_indices),
                                name = "Quartet_Position")
        self.model.addConstrs((Q[i, j] + (1 - Q[i + 1, j - 1]) >= 2*L[i, j] for (i, j) in ij_indices),
                                name = "Quartet_Position") # changed indices because Q asn't defined for their range
        self.model.addConstrs((Q[n - 1, j] + (1 - Q[0, j - 1]) >= 2 * L[n - 1, j] for j in range(2, n - 1)),
                                name = "Quartet_Position") # changed again
        
        self.model.addConstrs((P[i, j] + L[i, j] <= 1 for (i, j) in ij_indices),
                                name = "Quartet_Position")
        self.model.addConstrs((P[n - 1, j] + L[n - 1, j] <= 1 for j in nj_j_indices),
                                name = "Quartet_Position")
        self.model.addConstrs((P[i, j] + L[j, i] <= 1 for (i, j) in ij_indices),
                                name = "Quartet_Position")
        self.model.addConstrs((P[n - 1, j] + L[j, n-1] <= 1 for j in nj_j_indices),
                                name = "Quartet_Position")

        self.model.addConstrs((P[i, j] == 0 for (i, j) in list(set(itertools.product(range(n), range(n))) - set().union(ij_indices, nj_indices))))
        self.model.addConstrs((P[i, j] == 0 for (i, j) in list(set(itertools.product(range(n), range(n))) - set().union(ij_indices, nj_indices))))

class FullModel(FoldingModel):
    def __init__(self, nuc_sequence, name):
        # Notice this also requires a min. dist of 3
        super().__init__(nuc_sequence, name, 3)
        self.ij_indices = [(i, j) for i in range(0, self.n - 3) for j in range(i+3, self.n)]
        self.ij_indicesP = [(i, j) for i in range(1, self.n - 3) for j in range(i + 3, self.n - 1)]
        self.nj_j_indices = [j for j in range(2, self.n - 1)]
        self.nj_indices = [(self.n - 1, j) for j in self.nj_j_indices]
        self.quartet_indices = self.ij_indices + self.nj_indices
        self.Q = self.model.addVars([(i, j) for i in range(self.n) for j in range(self.n)], name = "Q", vtype = GRB.BINARY)
        self.P = self.model.addVars([(i, j) for i in range(self.n) for j in range(self.n)], name = "P", vtype = GRB.BINARY)
        self.L = self.model.addVars([(i, j) for i in range(self.n) for j in range(self.n)], name = "L", vtype = GRB.BINARY)

    def construct_constraints(self, num_crosses):
        self.single_match_constraints()
        self.min_dist_constraint()
        self.add_quartet_constraints()
        self.crossing_constraint(num_crosses)

    def declare_objective(self):
        nj_j_indices = self.nj_j_indices
        quartet_indices = self.quartet_indices
        Q = self.Q
        P = self.P
        L = self.L
        n = self.n
        ns = self.ns
        weight_dict = {
            ("a", "u") : 2,
            ("u", "a") : 2,
            ("g", "c") : 3,
            ("c", "g") : 3,
            ("a", "c") : 0.05,
            ("c", "a") : 0.05,
            ("g", "u") : 0.1,
            ("u", "g") : 0.1
        }
        f = {(i, j) : (weight_dict[self.ns[i], self.ns[j]] if (self.ns[i], self.ns[j]) in weight_dict.keys() else 0)
                        for (i, j) in self.possible_pairs}
        weight_dict = {
            ('a', 'u', 'a', 'u') : 9,
            ('a', 'u', 'c', 'g') : 21,
            ('a', 'u', 'g', 'c') : 24,
            ('a', 'u', 'u', 'a') : 13,
            ('c', 'g', 'a', 'u') : 22,
            ('c', 'g', 'c', 'g') : 33,
            ('c', 'g', 'g', 'c') : 34,
            ('c', 'g', 'u', 'a') : 24,
            ('g', 'c', 'a', 'u') : 21,
            ('g', 'c', 'c', 'g') : 24,
            ('g', 'c', 'g', 'c') : 33,
            ('g', 'c', 'u', 'a') : 21,
            ('u', 'a', 'a', 'u') : 12,
            ('u', 'a', 'c', 'g') : 23,
            ('u', 'a', 'g', 'c') : 21,
            ('u', 'a', 'u', 'a') : 16,
        }
        w = {(i, j) : (weight_dict[ns[i], ns[j], ns[i + 1], ns[j - 1]]
                        if (ns[i], ns[j], ns[i + 1], ns[j - 1]) in weight_dict.keys() else 1)
                for (i, j) in self.ij_indices}
        wn = {(n - 1, j) : (weight_dict[ns[n - 1], ns[j], ns[0], ns[j - 1]]
                            if (ns[n - 1], ns[j], ns[0], ns[j - 1]) in weight_dict.keys() else 1)
                for j in nj_j_indices}
        w.update(wn)
        self.model.setObjective(
            (quicksum([f[i, j] * self.x[i, j] for (i, j) in self.possible_pairs]) )
            + (quicksum([w[i, j]*(Q[i, j] + P[i, j] + L[i, j]) for (i, j) in quartet_indices])),
            GRB.MAXIMIZE
        )

    
    def add_quartet_constraints(self):
        n = self.n
        ij_indices = self.ij_indices
        ij_indicesP = self.ij_indicesP
        nj_j_indices = self.nj_j_indices
        nj_indices = self.nj_indices
        Q = self.Q
        P = self.P
        L = self.L

        # Basic Quartet Constraints
        self.model.addConstrs( (2 * Q[i, j] <= self.x[i, j] + self.x[i + 1, j - 1] for (i, j) in ij_indices),
                                name = "Quartet_Constr." )
        self.model.addConstrs( (2*Q[self.n - 1, j] <= self.x[j, self.n-1] + self.x[0, j - 1] for j in nj_j_indices),
                                    name = "Quartet_Constr." )
        self.model.addConstrs( (Q[i, j] == 0 for i, j in list(set(itertools.product(range(self.n), range(self.n))) - set().union(ij_indices, nj_indices))), name = "Quartet_Constr." )

        # Position of Quartet
        self.model.addConstrs((Q[i, j] >= P[i, j] for (i, j) in ij_indices), name = "Quartet_Position")
        self.model.addConstrs((Q[i, j] + (1 - Q[i - 1, j + 1]) >= 2*P[i, j] for (i, j) in ij_indicesP),
                                name = "Quartet_Position")
        self.model.addConstrs((Q[i, n - 1] + (1 - Q[0, i - 1]) >= 2*P[i, n - 1] for i in range(1, n-3)),
                                name = "Quartet_Position") # changed this one because Q wasn't defined for their range
        self.model.addConstrs((Q[0, j] + (1 - Q[n - 1, j + 1]) >= 2 * P[0, j] for j in range(3, n - 2)),
                                name = "Quartet_Position")
        self.model.addConstrs((Q[n - 1, j] >= P[n - 1, j] for j in range(2, n - 1)),
                                name = "Quartet_Position")
        self.model.addConstrs((Q[n - 1, j] + (1 - Q[n - 2, j + 1]) >= 2 * P[n - 1, j] for j in range(2, n - 1)),
                                name = "Quartet_Position") # changed to - 5 instead of - 3
        
        self.model.addConstrs((Q[i, j] >= L[i, j] for (i, j) in ij_indices),
                                name = "Quartet_Position")
        self.model.addConstrs((Q[i, j] + (1 - Q[i + 1, j - 1]) >= 2*L[i, j] for (i, j) in ij_indices),
                                name = "Quartet_Position") # changed indices because Q asn't defined for their range
        self.model.addConstrs((Q[n - 1, j] + (1 - Q[0, j - 1]) >= 2 * L[n - 1, j] for j in range(2, n - 1)),
                                name = "Quartet_Position") # changed again
        
        self.model.addConstrs((P[i, j] + L[i, j] <= 1 for (i, j) in ij_indices),
                                name = "Quartet_Position")
        self.model.addConstrs((P[n - 1, j] + L[n - 1, j] <= 1 for j in nj_j_indices),
                                name = "Quartet_Position")
        self.model.addConstrs((P[i, j] + L[j, i] <= 1 for (i, j) in ij_indices),
                                name = "Quartet_Position")
        self.model.addConstrs((P[n - 1, j] + L[j, n-1] <= 1 for j in nj_j_indices),
                                name = "Quartet_Position")

        self.model.addConstrs((P[i, j] == 0 for (i, j) in list(set(itertools.product(range(n), range(n))) - set().union(ij_indices, nj_indices))))
        self.model.addConstrs((P[i, j] == 0 for (i, j) in list(set(itertools.product(range(n), range(n))) - set().union(ij_indices, nj_indices))))        

if __name__ == '__main__':
    
    # Lengths 20, 30, 50, 100, 200 respectively
    test_strings = [
    'CGUCUUCACUACAGCAUCGG',
    'GACCUUACUGGGUACGAUUUACUGGAGGAC',
    'GGCCAGACUGGUGGUGUGACUCCAGGCUAACCGGAUACGCGUGCCUCGGG',
    'UAUGCAGGUCGCGUUUUUCCACUGCCUAGAUAGCUCUGAGGGUACACUUAGUUCAGCACAUAAGAGGGAUCAUACUAGGUCCGCGUCUUACCUCCUACGA'
    ]

    '''
    Models:
    - Crude
    - Simple Enhanced
    - Quartet
    - Crude + 10 crossings allowed
    - Simple + 10 crossings allowed
    - Quartet + 10 crossings allowed
    - Full: Quartet + Simple Enhancements + 10 crossings
    '''
    data = []
    for t in test_strings:
        print("BEGIN OF ITERATION LENGTH: ", len(t))
        c_model = CrudeModel(t, 'crude')
        s_model = SimpleEnhancedModel(t, 'simple')
        q_model = QuartetModel(t, 'quartet')
        cc_model = CrudeModel(t, 'crude_cross')
        sc_model = SimpleEnhancedModel(t, 'simple_cross')
        qc_model = QuartetModel(t, 'quartet_cross')
        full_model = FullModel(t, 'full')

        # We specify crossing using a passed in parameter
        print("Building c_model")
        c_model.build_model(0)
        print("Finished building c_model")
        print("Building s_model")
        s_model.build_model(0)
        print("Finished building s_model")
        print("Building q_model")
        q_model.build_model(0)
        print("Finished building q_model")
        print("Building cc_model")
        # Will now allows 10 crossings in the following models
        cc_model.build_model(10)
        print("Finished building cc_model")
        print("Building sc_model")
        sc_model.build_model(10)
        print("Finished building sc_model")
        print("Building qc_model")
        qc_model.build_model(10)
        print("Finished building qc_model")
        print("Building full_model")
        full_model.build_model(10)
        print("Finished building full_model")

        print("Solving c_model")
        c_model.solve_model()
        name = "c_model_%s.sol" % (len(t))
        c_model.model.write(name)
        print("Finished solving c_model")
        print("Solving s_model")
        s_model.solve_model()
        name = "s_model_%s.sol" % (len(t))
        s_model.model.write(name)
        print("Finished solving s_model")
        print("Solving q_model")
        q_model.solve_model()
        name = "q_model_%s.sol" % (len(t))
        q_model.model.write(name)
        print("Finished solving q_model")
        print("Solving cc_model")
        cc_model.solve_model()
        name = "cc_model_%s.sol" % (len(t))
        cc_model.model.write(name)
        print("Finished solving cc_model")
        print("Solving sc_model")
        sc_model.solve_model()
        name = "sc_model_%s.sol" % (len(t))
        sc_model.model.write(name)
        print("Finished solving sc_model")
        print("Solving qc_model")
        qc_model.solve_model()
        name = "qc_model_%s.sol" % (len(t))
        qc_model.model.write(name)
        print("Finished solving qc_model")
        print("Solving full_model")
        full_model.solve_model()
        name = "full_model_%s.sol" % (len(t))
        full_model.model.write(name)
        print("Finished solving full_model")

        loop_data = {
            'Model_Type' : ['Crude', 'Simple.E', 'Quartet', "Crude_Cross", 'Simple.E_Cross', 'Quartet_Cross', 'Full'],
            'ObjVal' : [c_model.model.ObjVal, s_model.model.ObjVal, q_model.model.ObjVal, cc_model.model.ObjVal,
                        sc_model.model.ObjVal, qc_model.model.ObjVal, full_model.model.ObjVal],
            'ObjBound' : [c_model.model.ObjBound, s_model.model.ObjBound, q_model.model.ObjBound, cc_model.model.ObjBound,
                        sc_model.model.ObjBound, qc_model.model.ObjBound, full_model.model.ObjBound],
            'MIPGap' : [c_model.model.MIPGap, s_model.model.MIPGap, q_model.model.MIPGap, cc_model.model.MIPGap,
                        sc_model.model.MIPGap, qc_model.model.MIPGap, full_model.model.MIPGap],
            'InputSize' : [len(t), len(t), len(t), len(t), len(t), len(t), len(t)],
            'ModelSize' : [c_model.model_size(), s_model.model_size(), q_model.model_size(), cc_model.model_size(),
                        sc_model.model_size(), qc_model.model_size(), full_model.model_size()],
            'RunTime' : [c_model.model.Runtime, s_model.model.Runtime, q_model.model.Runtime, cc_model.model.Runtime,
                        sc_model.model.Runtime, qc_model.model.Runtime, full_model.model.Runtime],
            'Work' : [c_model.model.Work, s_model.model.Work, q_model.model.Work, cc_model.model.Work,
                        sc_model.model.Work, qc_model.model.Work, full_model.model.Work]
        }
        data.append(loop_data) 
        print("End OF ITERATION LENGTH: ", len(t))

    result = pd.DataFrame(data = data[0])
    for i in range(1, len(data)):
        temp_df = pd.DataFrame(data = data[i])
        result = pd.concat([result, temp_df], axis=0)
    
    print(result)

    fig = px.line(data_frame= result, x = "InputSize", y = "RunTime", color= "Model_Type")
    fig.show()

    fig = px.line(data_frame= result, x = "InputSize", y = "Work", color= "Model_Type")
    fig.show()

    fig = px.line(data_frame= result, x = "InputSize", y = "ModelSize", color= "Model_Type")
    fig.show()

    crude_model = CrudeModel('CGUCUUCACUACAGCAUCGG', 'crude')
    crude_model.build_model(0)
    crude_model.solve_model()
    print("Obj: ", crude_model.model.ObjVal)
    print("Size: ", crude_model.model_size())
    print("Vars:")
    for v in crude_model.model.getVars():
        if (v.x == 1):
            print("%s = %g" % (v.varName, v.x))

    full_model = FullModel('CGUCUUCACUACAGCAUCGG', 'full')
    full_model.build_model(10)
    full_model.solve_model()
    print("Obj: ", full_model.model.ObjVal)
    print("Size: ", full_model.model_size())
    print("Vars:")
    for v in full_model.model.getVars():
        if (v.x == 1):
            print("%s = %g" % (v.varName, v.x))
    result.to_csv("full_model_output_reg_python.csv", index = True)
    # results.to_csv("/Users/q3/Desktop/GT/Courses/Active\ Courses/ISYE\ 3133/Project/full_model_output.csv", index = True)