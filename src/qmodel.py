class QModel:
    def __init__(self):
        pass

    def actions(self, numerical_state: dict[str, float] = None) -> list[dict[str, int]]:
        """
        Implement the list of all possible actions. The given numerical_state can be used to
        determine the operating region if the list depends on the region.
        """
        return []

    def effect(self, qualitative_action: dict[str, int], numerical_state: dict[str, float] = None) -> dict[str, int]:
        """
        Implement a mapping from the given qualitative action to the vector of qualitative effects.
        The numerical_state is used when multiple operating regions are defined. If only one
        operating region exists within the model, the numerical_state parameter can be ignored.

        qualitative_action is a dictionary of control variable names and their qualitative directions, e.g.:
        qualitative_action = {'m1':1, 'm2':0, 'm3':-1, ...},

        numerical_state is a dictionary of system variables and their numerical values, e.g.:
        numerical_state = {'x':0.23, 'y':-2.34, ...}

        The return value is a dictionary of goal variable names and their qualitative directions, e.g.:
        {'x':1, 'y':-1, ...}. Let non-deterministic effects have any value outside the {-1, 0, 1} set.
        """
        return {}