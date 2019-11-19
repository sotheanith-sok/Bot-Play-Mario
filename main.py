import retro
from gym.wrappers import Monitor
import numpy as np
import cv2
from DDQAgent import DDQAgent

# Build game enviroment
env = retro.make(game="SuperMarioWorld-Snes", state="YoshiIsland1")

# Recoder video for every 10 episodes
env = Monitor(env, './video', force=True, video_callable=lambda episode_id: episode_id%10==0)

# How many episodes to play
n_games = 5000

# Initialize double deep q agent
agent = DDQAgent(
    alpha=0.005,       # Learning Rate
    gamma=0.999,         # Discount factor. Make future event weighted less
    n_actions=6,      # Number of possible actions. 2^8 for 8 inputs
    epsilon=1.0,        # How often should agent "explore" (Do random action). Set to 0 for well train model
    epsilon_dec=0.9999,  #How fast should start perform greedy action
    batch_size=64,      # How many samples should this agent train on
    input_dimension=(56, 64, 3),  # Input dimension.
    memory_size=5000,   # Max capacity of ReplayBuffer
)

#Load agent
agent.load_model()

#Keep track of scores
scores = []

#Some variable
learnEvery = 30         #Keep track of how many frame between each time agent learn
rememberEvery = 5       #How many frame between each time agent remember
frame_skip=5            #Only getting new action every 5 frame

#Start playing
for i in range(n_games):

    done = False 
    score = 0
    oberservation = cv2.resize(env.reset() / 255.0,(64,56)) #Scale rgb to between 0 and 1 and resize frame
    frame_counter = 0

    # action = agent.choose_action(oberservation)
    while not done:
        #Get new actions per skip
        if frame_counter%frame_skip==0:
            action = agent.choose_action(oberservation)

        # Time Step
        new_oberservation, reward, done, _ = env.step(action)
        #Scale rgb to between 0 and 1 and resize frame
        new_oberservation = cv2.resize(new_oberservation / 255.0,(64,56))
        score += reward

        
        # Agent only remember 1 frame for every 10 frame
        if frame_counter % rememberEvery == 0:
            agent.remember(oberservation, action, reward, new_oberservation, done)

        oberservation = new_oberservation
        # Agent will learn every 300 frame
        if frame_counter % learnEvery == 0:
            agent.learn()
        
        frame_counter+=1
    
    scores.append(score)
    avg_score = np.mean(scores[max(0, i - 100) : i + 1])
    print(
        "Episode",
        i,
        "score %.2f" % score,
        "Average score %.2f" % avg_score,
        "Epsilon %.2f" % agent.epsilon,
    )

    #Save model every 10 episodes
    if i % 10 == 0 and i > 0:
        agent.save_model()
