import random
import numpy as np

class Creature:
	def __init__(self, neural_network, max_x, max_y, initial_energy=50):
		"""
		Initialize a creature at a random position and orientation
		with random speed and given neural network, initial energy and max x and y.

		Parameters
		----------
		neural_network : NeuralNetwork
			The neural network to use for decision making.
		max_x : int
			Maximum x coordinate.
		max_y : int
			Maximum y coordinate.
		initial_energy : int, optional
			Initial energy of the creature. Defaults to 50.
		"""
		self.x = random.randint(0, max_x)
		self.y = random.randint(0, max_y)
		self.speed = random.uniform(0, 10)
		self.energy = initial_energy
		self.neural_network = neural_network
		self.orientation = random.uniform(0, 2 * np.pi)
		self.closest_food_id = None
		self.score = 0

  
class Food:
	def __init__(self, food_id, width, height):
		"""
		Initialize a food item with a given id at a random position within a given
  		width and height.

		Parameters
		----------
		food_id : int
			The id of the food item.
		width : int
			The maximum x coordinate.
		height : int
			The maximum y coordinate.
		"""
		self.id = food_id
		self.x = random.randint(0, width)
		self.y = random.randint(0, height)
