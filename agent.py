import numpy as np

class Agent:
    def __init__(self, id, loc, mind):
        self.id = id
        self.alive = True
        self.loc = loc
        self.current_state = None
        self.action = None
        self.next_state = None
        self.mind = mind
        self.input_size = mind.get_input_size()
        self.output_size = mind.get_output_size()
        self.age = 0
        self.decision = None
        self.divided = False

    def update(self, reward, done):
        assert self.action != None, 'No Action'
        assert reward != None, 'No Reward'
        self.mind.remember([[[self.current_state]], [self.current_state.sum()/(self.current_state.shape[0] * self.current_state.shape[1])], 
                            [self.action], [[self.next_state]], [reward], [done]])
        #print([[[self.current_state]], [self.action], [[self.next_state]], [reward], [done]])

        #loss = self.mind.train()

        self.action = None
        if not done:
            self.current_state, self.next_state = self.next_state, None
        else:
            self.current_state, self.next_state = None, None

    def get_losses(self):
        return self.mind.get_losses()

    def decide(self, state):
        #print(state)
        self.action = self.mind.decide(state, state.sum()/(state.shape[0] * state.shape[1]))  # need to edit!!
        self.age += 1
        return self.action

    def get_state(self):
        return self.current_state
    
    def get_age(self):
        return self.age

    def get_id(self):
        return self.id

    def get_loc(self):
        return self.loc

    def get_decision(self):
        assert self.decision != None, "Decision is requested without setting."
        return self.decision

    def set_decision(self, decision):
        self.decision = decision

    def clear_decision(self):
        self.decision = None

    def set_loc(self, loc):
        self.loc = loc

    def set_current_state(self, state):
        self.current_state = state

    def set_next_state(self, state):
        self.next_state = state

    def is_alive(self):
        return self.alive

    def respawn(self, loc):
        self.alive = True
        self.clear_decision()
        self.set_loc(loc)
