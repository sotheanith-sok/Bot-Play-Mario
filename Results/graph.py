import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

file = open("./performance_file.bin", "rb")
array = []
while 1:
    try:
        b = pickle.load(file)
        array.append([b[2], b[0]])
    except (EOFError):
        break

array = np.array(array)
data = pd.DataFrame(array, columns=["epsilon", "scores"])
# new_data = []
# for name_of_the_group, group in data.groupby("epsilon"):
#     new_data.append([group.mean()[0], group.mean()[1]])

# new_data = np.array(new_data)
new_data = pd.DataFrame(data, columns=["epsilon", "scores"])
plt.ylim(0, 1000000)

ax = sns.lineplot(x="epsilon", y="scores", data=new_data, estimator=np.mean)
ax.invert_xaxis()
plt.show()
