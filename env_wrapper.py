from unityagents import UnityEnvironment


class EnvWrapper:

    def __init__(self, file_name='Reacher_Windows_x86_64\Reacher.exe', train_mode=True):
        self.env = UnityEnvironment(file_name)
        self.brain_name = self.env.brain_names[0]
        self.train_mode = train_mode

    def reset(self):
        env_info = self.env.reset(train_mode=self.train_mode)[self.brain_name]
        return env_info.vector_observations

    def step(self, actions):
        env_info = self.env.step(actions)[self.brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        return rewards, next_states, dones

    def close(self):
        self.env.close()