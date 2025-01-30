import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output

def get_run_df(creatures):
	df = pd.DataFrame([{
		'neural_network': creature.neural_network,
		'score': creature.score,
		'creature': creature
		} for creature in creatures])
	return df.sort_values(by='score', ascending=False)


def plot_state(creatures, foods, x_width, y_width):
	clear_output(wait=True)
	fig = plt.figure(figsize=(5,5))
	ax = plt.axes()

	ax.set_xlim(0,x_width)
	ax.set_ylim(0,y_width)
	for i in creatures:
		ax.scatter(i.x, i.y, c='r')
		ax.plot([i.x, i.x + np.cos(i.orientation)*i.speed], [i.y,i.y+np.sin(i.orientation)*i.speed], c='k')
	
	
	for key, item in foods.items():
		ax.scatter(item.x,item.y, c='g')
	plt.pause(0.0001)
	plt.show()
