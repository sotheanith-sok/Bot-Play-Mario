import retro
import numpy as np
from DDQAgent import DDQAgent


env = retro.make(game="SuperMarioWorld-Snes", state="Forest1")
n_games = 6400
agent = DDQAgent(
    alpha=0.0005,
    gamma=0.99,
    n_actions=4096,
    epsilon=1.0,
    batch_size=64,
    input_dimension=(224, 256, 3),
    memory_size=1000
)

scores = []
render_counter = 64
for i in range(n_games):
    done = False
    score = 0
    oberservation = env.reset() /255.
    while not done:
        action = agent.choose_action(oberservation)
        new_oberservation, reward, done, _ = env.step(action)
        new_oberservation= new_oberservation/255.
        score += reward
        agent.remember(oberservation, action, reward, new_oberservation, done)
        oberservation=new_oberservation
        agent.learn()       
        # if i%render_counter==0:
        #   env.render()
    
    scores.append(score)
    avg_score =np.mean(scores[max(0,i-100):i+1])
    print('Episode',i, 'score %.2f' %score, 'Average score %.2f' %avg_score, "Epsilon %.2f" %agent.epsilon)

    if i%10==0 and i >0:
        agent.save_model()
        
