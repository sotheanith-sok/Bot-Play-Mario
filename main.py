import retro
from gym.wrappers import Monitor
import numpy as np
from DDQAgent import DDQAgent


env = retro.make(game="SuperMarioWorld-Snes", state="Forest1")
# env = Monitor(env, './video', force=True, video_callable=lambda episode_id: episode_id%2==0)
n_games = 5000
agent = DDQAgent(
    alpha=0.0005,
    gamma=0.99,
    n_actions=256,
    epsilon=1.0,
    batch_size=64,
    input_dimension=(224, 256, 3),
    memory_size=1000,
)

agent.load_model()

scores = []
render_counter = 10
learning_counter = 0
remember_counter = 0
for i in range(n_games):
    done = False
    score = 0
    oberservation = env.reset() / 255.0
    while not done:
        action = agent.choose_action(oberservation)
        if i < 300 and np.random.random_sample()<0.8:
            action[7] = 1
        new_oberservation, reward, done, _ = env.step(action)
        new_oberservation = new_oberservation / 255.0
        score += reward
        oberservation = new_oberservation

        if remember_counter % 10 == 0:
            agent.remember(oberservation, action, reward, new_oberservation, done)

        if learning_counter % 300 == 0:
            agent.learn()

        # if i % render_counter == 0:
        #     env.render()

        learning_counter += 1

    scores.append(score)
    avg_score = np.mean(scores[max(0, i - 100) : i + 1])
    print(
        "Episode",
        i,
        "score %.2f" % score,
        "Average score %.2f" % avg_score,
        "Epsilon %.2f" % agent.epsilon,
    )

    if i % 10 == 0 and i > 0:
        agent.save_model()
