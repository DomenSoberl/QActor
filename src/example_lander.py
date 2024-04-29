import gymnasium as gym
from qmodel import QModel
from qactor import QActor

class LanderModel(QModel):
    def actions(self, numerical_state: dict[str, float] = None) -> list[dict[str, int]]:
        return [
            {'m1': 0, 'm2': 0, 'm3': 0}, # do nothing
            {'m1': 1, 'm2': 0, 'm3': 0}, # left engine
            {'m1': 0, 'm2': 1, 'm3': 0}, # main engine
            {'m1': 0, 'm2': 0, 'm3': 1}, # right engine
        ]

    def effect(self, qualitative_action: dict[str, int], numerical_state: dict[str, float] = None) -> dict[str, int]:
        """
        We use the following qualitative model:
            ax = M-,+(m1, m3)
            ay = M+(m2)
            ar = M+,-(m1, m3)
            
            deriv(x, vx)
            deriv(vx, ax)
            deriv(y, vy)
            deriv(vy, ay)
            deriv(r, vr)
            deriv(vr, ar)
        """

        m1 = qualitative_action['m1']
        m2 = qualitative_action['m2']
        m3 = qualitative_action['m3']

        # Resolve: ax = M-,+(m1, m3).
        if m1 == 0 and m3 == 0:
            ax = 0
        elif m1 == 0:
            ax = m3
        elif m3 == 0:
            ax = -m1
        elif m1 < 0 and m3 > 0:
            ax = 1
        elif m1 > 0 and m3 < 0:
            ax = -1
        else:
            ax = -2 # non-deterministic
        
        # Resolve: ar = M+,-(m1, m3).
        if m1 == 0 and m3 == 0:
            ar = 0
        elif m1 == 0:
            ar = -m3
        elif m3 == 0:
            ar = m1
        elif m1 < 0 and m3 > 0:
            ar = -1
        elif m1 > 0 and m3 < 0:
            ar = +1
        else:
            ar = -2 # non-deterministic

        # Resolve: ay = M+(m2).
        ay = m2

        # Propagate relative qualitative effects.
        return {'x': ax, 'vx': ax, 'y': ay, 'vy': ay, 'r': ar, 'vr': ar}

def run_episode(env: gym.Env, actor: QActor) -> float:
    # Restart the actor, but keep the learned parameters.
    actor.restart()
    
    # Reset the environment to the initial state.
    [x, y, vx, vy, r, vr, _, _], _ = env.reset()
    state = {'x': x, 'y': y, 'r': r, 'vx': vx, 'vy': vy,'vr': vr}

    # Run one episode and collect the reward.
    step = 0
    episode_terminated = False
    episode_reward = 0
    while not episode_terminated:
        # Observe the current numerical state.
        actor.observe(numerical_state=state, dt=0.02)
        
        # Regulate the horizontal position by moving the target speed.
        if x > 0:
            target_vx = -0.1
        elif x < 0:
            target_vx = 0.1
        else:
            target_vx = 0

        # Decide on the next qualitative action.
        qualitative_action = actor.act(target={'r': 0, 'vx': target_vx, 'vy': -0.1, 'vr': 0})

        # Translate the qualitative action to the environment output.
        if qualitative_action['m1'] == 1:
            action = 1
        elif qualitative_action['m2'] == 1:
            action = 2
        elif qualitative_action['m3'] == 1:
            action = 3
        else:
            action = 0

        # Execute the action and get the next state.
        [x, y, vx, vy, r, vr, t1, t2], reward, terminated, truncated, _ = env.step(action)
        state = {'x': x, 'y': y, 'vx': vx, 'vy': vy, 'r': r, 'vr': vr}

        # Collect the reward.
        episode_reward += reward

        # Does the episode end?
        episode_terminated = (t1 and t2) or terminated or truncated

        # Count steps.
        step += 1

    return episode_reward

# Create the environment and the qualitative actor.
env = gym.make("LunarLander-v2", render_mode="human")
actor = QActor(LanderModel())

# Run 10 episodes and print the cumulative rewards.
for episode in range(10):
    reward = run_episode(env, actor)
    print(episode, reward)
    
env.close()