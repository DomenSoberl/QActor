import math
from qmodel import QModel

class QActor:
    def __init__(self, model: QModel):
        self._model = model
        self.reset()
    
    """
    Clear the current state and the learned parameters.
    """
    def reset(self):
        self._current_x = {}
        self._current_v = {}
        self._current_a = {}

        self._max_v_pos = {}
        self._max_v_neg = {}
        self._max_a_pos = {}
        self._max_a_neg = {}

    """
    Clear the current state, but keep the learned parameters.
    """
    def restart(self):
        self._current_x = {}
        self._current_v = {}
        self._current_a = {}

    """
    Store the current numerical state. Infer velocities and accelerations.
    """
    def observe(self, numerical_state: dict[str, float], dt: float):
        for variable in numerical_state:
            x = numerical_state[variable]
            v = None
            a = None

            # Observe the current value and compute the velocity.
            if variable in self._current_x:
                x0 = self._current_x[variable]
                v = (x - x0) / dt
            self._current_x[variable] = x
            
            # Observe the current velocity and compute the acceleration.
            if v is not None:
                if variable in self._current_v:
                    v0 = self._current_v[variable]
                    a = (v - v0) / dt
                self._current_v[variable] = v

                # Is the current velocity maximal of all observed by now?
                if v != 0:
                    max_v = self._max_v_pos if v > 0 else self._max_v_neg
                    if variable in max_v:
                        if abs(v) > max_v[variable]:
                            max_v[variable] = abs(v)
                    else:
                        max_v[variable] = abs(v)
            
            # Observe the current acceleration.
            if a is not None:
                self._current_a[variable] = a # Not really needed, but we store it anyway.

                # Is the current acceleration maximal of all observed by now?
                if a != 0:
                    max_a = self._max_a_pos if a > 0 else self._max_a_neg
                    if variable in max_a:
                        if abs(a) > max_a[variable]:
                            max_a[variable] = abs(a)
                    else:
                        max_a[variable] = abs(a)
    
    """
    Decide which qualitative action to take in the current state to reach the given target.
    """
    def act(self, target: dict[str, float]) -> dict[str, int]:
        # Store votes as tupples (action: dict[str, int], vote: float, rank: int).
        votes = []

        # Evaluate all available actions.
        for action in self._model.actions(numerical_state=self._current_x):
            action_effect = self._model.effect(qualitative_action=action, numerical_state=self._current_x)

            # Let target variables vote for this action.
            action_vote = 0
            for variable in target:
                current_value = self._current_x[variable]
                target_value = target[variable]
                
                desired_direction = math.copysign(1, target_value - current_value)
                action_direction = action_effect[variable] if abs(action_effect[variable]) <= 1 else 0

                eta = self._eta(variable, target[variable]) # The estimated time of arrival (ETA).
                action_vote += (desired_direction * action_direction) * eta

            # Compute the rank of the action (the number of its deterministic non-zero values).
            action_rank = 0
            for variable in action:
                action_rank += 1 if abs(action[variable]) == 1 else 0
            
            # Store the vote for this action.
            votes.append((action, action_vote, action_rank))
        
        # Sort actions by their votes and ranks.
        votes = sorted(votes, key=lambda x: (x[1], x[2]), reverse=True)

        # Return the action with the highest vote and rank.
        return votes[0][0]

    def _eta(self, variable: str, x1: float) -> float:
        # Presume motion in the positive direction.
        x = self._current_x[variable]
        v0 = self._current_v[variable] if variable in self._current_v else 0
        v1_pos = self._max_v_pos[variable] if variable in self._max_v_pos else 1
        v1_neg = self._max_v_neg[variable] if variable in self._max_v_neg else 1
        acc = self._max_a_pos[variable] if variable in self._max_a_pos else 1
        dcc = self._max_a_neg[variable] if variable in self._max_a_neg else 1

        # Distance to be made towards the goal
        dist = x1 - x

        # If motion must be done in the presumed positive direction.
        if dist > 0:
            return QActor._time_to_goal(dist, v0, v1_pos, acc, dcc)
        
        # If motion must be done in the negative direction, flip the directions.
        if dist < 0:
            return QActor._time_to_goal(-dist, -v0, v1_neg, dcc, acc)
        
        # If the target has been reached.
        return 0

    def _time_to_goal(dist: float, v0: float, v1: float, acc: float, dcc: float) -> float:
        """
        dist - distance to be made (unsigned).
        v0   - current speed (positive, if towards the goal).
        v1   - maximum speed (unsigned).
        acc  - acceleration (towards the goal, unsigned).
        dcc  - deceleration (acceleration away from the goal, unsigned).
        """

        t = 0 # Time to reach the goal.

        # If v is non-negative, we use equation (v1)^2 = v^2 + 2a(dx).
        if v0 >= 0: # We are moving towards the goal.
            vx = math.sqrt(v0**2 + 2 * acc * dist) # The final speed that will be reached at the goal position.
            if vx <= v1:                           # If the final speed does not exceed the maximum speed,
                t += (vx - v0) / acc               # compute the time to reach the final speed.
            else: # If the final speed exceeds the maximum speed,
                t += (v1 - v0) / acc;              # First, compute the time to reach the maximum speed.
                s = (v1**2 - v0**2) / (2 * acc)    # Displacement made by the time maximum speed is reached.
                t += (dist - s) / v1               # The rest of the distance is made with the maximum speed.
            return t

        # if v is negative, we first compute the stopping point.
        else: # We are moving away from the goal.
            s = v0**2 / (2 * acc)                                # Displacement made until stopped.
            t += -v0 / acc                                       # Time until stopped.
            t += QActor._time_to_goal(dist + s, 0, v1, acc, dcc) # Time to reach the goal from the stopped position.
            return t