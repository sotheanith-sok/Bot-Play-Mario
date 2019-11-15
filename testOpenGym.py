import retro

env = retro.make(game="SuperMarioWorld-Snes", state="Bridges1")
obs = env.reset()
while True:
    obs, rew, done, info = env.step(env.action_space.sample())
    env.render()
    if done:
        obs = env.reset()
env.close()
