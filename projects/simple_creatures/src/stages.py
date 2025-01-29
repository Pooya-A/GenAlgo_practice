import numpy as np
import torch

def get_creatures_closets_food(creatures, foods, x_width, y_width):
    """
    Assigns the closest food item to each creature based on the Euclidean distance.

    For each creature with non-zero energy, this function calculates the distance
    to all available food items and assigns the closest food item to the creature.
    If a creature has zero energy, its closest food attribute is set to None.

    Parameters
    ----------
    creatures: A list of Creature objects
        Each creature has attributes such as position (x, y) and energy level.
    foods: A dictionary of Food objects
        The keys are identifiers for the food items, and the values are Food objects
        with attributes such as position (x, y).
    x_width: int
        The width of the area in which creatures and food exist.
    y_width: int
        The height of the area in which creatures and food exist.

    Returns
    -------
    creatures: A list of Creature objects
        The list of creatures with updated closest food attributes.
    """

    for creature in creatures:
        min_dist = np.sqrt(x_width**2 + y_width**2)
        if creature.energy == 0:
            creature.closest_food = None
            continue
        creature.closest_food = None
        for key,food in foods.items():
            dist = np.sqrt((creature.x - food.x)**2 + (creature.y - food.y)**2)
            if dist < min_dist:
                min_dist = dist
                creature.closest_food = key
    return creatures


def move_creatures(creatures, x_max, y_max):
    """
    Move each creature in the list according to its orientation and speed.

    Each creature with non-zero energy moves according to its orientation and speed.
    If the new position is outside the specified x and y max, the creature is moved
    to the other side of the boundary. The energy level of each creature is
    reduced by one.

    Parameters
    ----------
    creatures: A list of Creature objects
        Each creature has attributes such as position (x, y), energy level, orientation,
        and speed.
    x_max: int
        The width of the area in which creatures exist.
    y_max: int
        The height of the area in which creatures exist.

    Returns
    -------
    creatures: A list of Creature objects
        The list of creatures with updated positions and energy levels.
    """
    for creature in creatures:
        if creature.energy == 0:
            creature.speed = 0
            continue
        creature.energy -= 1
        new_x = creature.x + np.cos(creature.orientation) * creature.speed
        new_y = creature.y + np.sin(creature.orientation) * creature.speed
        if new_x < 0:
            new_x += x_max
        if new_y < 0:
            new_y += y_max
        if new_x > x_max:
            new_x -= x_max
        if new_y > y_max:
            new_y -= y_max
        creature.x = new_x
        creature.y = new_y
    return creatures


def creatures_actions(creatures, foods):
    """
    For each creature, checks if it is near its closest food and if so eats it and
    increments score. Then increments score for all creatures.

    Parameters
    ----------
    creatures: A list of Creature objects
    foods: A dictionary of Food objects
    Returns
    -------
    creatures: The list of creatures
    """
    for creature in creatures:
        if creature.energy == 0:
            continue
        if creature.closest_food in foods.keys():
            food = foods[creature.closest_food]
            if (creature.x - food.x)**2 + (creature.y - food.y)**2 <= 25:
                creature.energy += 30
                foods.pop(creature.closest_food)
        creature.score += 1
    return creatures


def get_creature_inputs(creature, foods):
    """
    Returns the input values for the creature. The values are the distance and
    relative angle to the closest food.

    Parameters
    ----------
    creature: A Creature object
    foods: A dictionary of Food objects

    Returns
    -------
    inputs: A list of two floats, the distance and angle
    """
    if not creature.closest_food:
        return [-1.0,-1.0]
    food = foods[creature.closest_food]
    dist = np.sqrt((creature.x - food.x)**2 + (creature.y - food.y)**2)
    dist = float(dist / 256)
    angle = np.arctan2(food.y - creature.y , food.x- creature.x)
    angle = float(creature.orientation - angle)
    return [dist, angle]


def update_creatures(creatures, foods):
    """
    Updates each creature in the list by running its neural network with the input
    values, and updating its speed and orientation.

    Parameters
    ----------
    creatures: A list of Creature objects
    foods: A dictionary of Food objects

    Returns
    -------
    creatures: The list of creatures
    """
    
    for creature in creatures:
        # Get the input values for the creature
        if creature.energy == 0:
            continue
        input_values = get_creature_inputs(creature, foods)

        # Run the neural network and get the outputs
        outputs = creature.neural_network(torch.tensor([input_values + [creature.speed]]))

        # Calculate the delta orientation
        delta_orientation = np.clip(
            np.pi * float(outputs[0][1]) * 2 - np.pi, -np.pi / 2, np.pi / 2
        )
        # Update the creature's speed
        creature.speed = float(np.clip(float(outputs[0][0]) * 10, 0, 10))

        # Update the creature's orientation
        creature.orientation -= delta_orientation
        if creature.orientation > 2 * np.pi:
            creature.orientation -= 2 * np.pi
        if creature.orientation < -2 * np.pi:
            creature.orientation += 2 * np.pi
    return creatures
