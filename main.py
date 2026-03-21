from src.wrappers.cluster_simulator.render_wrapper import ClusterGameRendererWrapper
import src
import gymnasium as gym
from src.envs.cluster_simulator.metric_based.renderer import ClusterMetricRenderer

def main():
    env = gym.make("ClusterScheduling-metric-online-v1", n_jobs=50, n_machines=10)
    render = ClusterMetricRenderer(render_mode="human", cell_size=20)
    env = ClusterGameRendererWrapper(env, render)
    max_steps = 2_000
    print(env)
    env.reset()
    for step in range(max_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

if __name__ == '__main__':
   main()