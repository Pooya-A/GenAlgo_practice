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


def plot_state(creatures, foods, width, height):
	"""
	Plots the current state of the creatures and foods.

	Parameters
	----------
	creatures: A list of Creature objects
	foods: A dictionary of Food objects
	width: The width of the grid
	height: The height of the grid
	"""
	clear_output(wait=True)
	fig, ax = plt.subplots(figsize=(5, 5))

	ax.set_xlim(0, width)
	ax.set_ylim(0, height)

	for creature in creatures:
		ax.scatter(creature.x, creature.y, color='red')
		ax.plot(
			[creature.x, creature.x + np.cos(creature.orientation) * creature.speed],
			[creature.y, creature.y + np.sin(creature.orientation) * creature.speed],
			color='black'
		)

	for food in foods.values():
		ax.scatter(food.x, food.y, color='green')

	plt.pause(0.0001)
	plt.show()
