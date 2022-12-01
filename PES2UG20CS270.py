import numpy as np


class HMM:
    """
    HMM model class
    Args:
        A: State transition matrix
        states: list of states
        emissions: list of observations
        B: Emmision probabilites
    """

    def __init__(self, A, states, emissions, pi, B):
        self.A = A
        self.B = B
        self.states = states
        self.emissions = emissions
        self.pi = pi
        self.N = len(states)
        self.M = len(emissions)
        self.make_states_dict()

    def make_states_dict(self):
        """
        Make dictionary mapping between states and indexes
        """
        self.states_dict = dict(zip(self.states, list(range(self.N))))
        self.emissions_dict = dict(
            zip(self.emissions, list(range(self.M))))

    def viterbi_algorithm(self, seq):
        """
        Function implementing the Viterbi algorithm
        Args:
            seq: Observation sequence (list of observations. must be in the emmissions dict)
        Returns:
            nu: Porbability of the hidden state at time t given an obeservation sequence
            hidden_states_sequence: Most likely state sequence 
        """
        # TODO
        T = len(seq)
        nu = np.zeros((T, self.N))
        temp = np.zeros((T, self.N), dtype=int)

        for i in range(self.N):
            temp[0, i] = 0
            nu[0, i] = self.pi[i] * self.B[i, self.emissions_dict[seq[0]]]
           
        for i in range(1, T):
            for j in range(self.N):
                maxNu = -1
                maxTemp = -1

                for k in range(self.N):
                    loaclaNu = nu[i - 1, k] * self.A[k, j] * \
                        self.B[j, self.emissions_dict[seq[i]]]
                    if loaclaNu > maxNu:
                        maxNu = loaclaNu
                        maxTemp = k
                nu[i, j] = maxNu
                temp[i, j] = maxTemp
        
        maxNu = -1
        maxTemp = -1

        for j in range(self.N):
            loaclaNu = nu[T - 1, j]
            if loaclaNu > maxNu:
                maxNu = loaclaNu
                maxTemp = j
        
        states = [maxTemp]
        for i in range(T - 1, 0, -1):
            states.append(temp[i, states[-1]])

        states.reverse()
        self.states_dict = {v: k for k, v in self.states_dict.items()}
        return [self.states_dict[i] for i in states]