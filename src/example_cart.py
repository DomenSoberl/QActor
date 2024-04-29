from qmodel import QModel
from qactor import QActor
import matplotlib.pyplot as plt

"""
A very simple 1-dimensional cart simulator.
"""
class CartSimulator():
    def __init__(self, mass, x0):
        self._mass = mass
        self._x0 = x0
        self.reset()
    
    def reset(self):
        self.x = self._x0
        self.v = 0
        self.t = 0
    
    def step(self, F, dt):
        a = F / self._mass
        self.v += a * dt
        self.x += self.v * dt
        self.t += dt

"""
Define the cart model using the QModel base class.
"""
class CartModel(QModel):
    def actions(self, numerical_state: dict[str, float] = None) -> list[dict[str, int]]:
        # There are 3 possible qualitative actions, F = dec/std/inc.
        return [
            {'F': -1},
            {'F': 0},
            {'F': 1},
        ]

    def effect(self, qualitative_action: dict[str, int], numerical_state: dict[str, float] = None) -> dict[str, int]:
        """
        We use the following qualitative model:
            a = M+(F)
            deriv(x, v)
            deriv(v, a)
        
        According to the definition of 'relative qualitative effects' (Soberl, 2017),
        a relative qualitative effect propagates across derivatives.
        """
        return {
            'x': qualitative_action['F'],
            'v': qualitative_action['F']
        }

"""
Runs an episode for the given number of steps and returns the time when the goal state was reached.
"""
def run_episode(episode: int, simulator: CartSimulator, actor: QActor, steps: int, dt: float) -> float:
    # Restarting the actor means clearing the current state but keeping the learned parameters.
    actor.restart()
    
    # Reset the simulator to its initial state.
    simulator.reset()

    # Start with zero force.
    F = 0
    goal_time = None
    
    # Plotting data.
    plot_t, plot_x, plot_v, plot_F = [simulator.t], [-simulator.x], [simulator.v], [F]

    # Simulate individual time steps.
    for _ in range(steps):
        # Let the actor observe the current numerical state, velocities and accelerations.
        actor.observe(numerical_state={'x': simulator.x, 'v': simulator.v}, dt=dt)
        
        # Let the actor decide on the qualitative action.
        action = actor.act(target={'x': 0, 'v': 0})

        # Change the current force in steps of 1.0 N.
        F += 1.0 * action['F']
        
        # Clip the numerical action to [-10, 10] N.
        if F > 10: F = 10
        if F < -10: F = -10

        # Execute the numerical action.
        simulator.step(F=F, dt=dt)

        # Check if the goal has been reached.
        if goal_time == None and simulator.x >= 0:
            goal_time = simulator.t

        # Store the plotting data.
        plot_t.append(simulator.t)
        plot_x.append(-simulator.x)
        plot_v.append(simulator.v)
        plot_F.append(F)

    # Plot the distance, velocity and the applied force.
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.set_ylabel('Distance (x) / Velocity (v)')
    ax1.set_xlabel('Time (t)')
    ax1.set_xlim(0, 4)
    ax1.set_ylim(0, 10)

    ax2.set_ylabel('Applied force (F)')
    ax2.set_xlim(0, int(steps*dt))
    ax2.set_ylim(-10, 10)

    ax1.axhline(y=0, color='black')
    ax1.plot(plot_t, plot_x, color='green')
    ax1.plot(plot_t, plot_v, color='red')
    ax2.plot(plot_t, plot_F, color='gold')    
    
    fig.savefig("plot{}.png".format(episode))

    return goal_time

# Create the actor and the simulator.
actor = QActor(CartModel())
simulator = CartSimulator(mass=1.0, x0=-10.0)

# Run 10 episodes and print out the times at which the goal has been reached.
for episode in range(10):
    goal_time = run_episode(episode=episode, simulator=simulator, actor=actor, steps=400, dt=0.01)
    print(episode, goal_time)


