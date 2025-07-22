# EVO-SYSTEM
Simulation of an evolving ecosystem using Perlin noise for procedural world generation in Python.

## World Generation
The world is generated using Perlin noise, implemented through the custom `noise` library. This library includes functions to create various environmental maps (height, temperature, humidity, etc.). A Gaussian mask can be applied to create an island-like shape in the center of the world. It also provides tools to visualize these maps. An example usage is included in the `main()` function.

## Organisms: Stats and Behavior
Organisms are defined in the custom `organism` library, which includes all the constants related to their needs, reproduction chances, and energy gain.

- **Plants** stats: height, roots, leaves and lifespan.
- **Animals** stats: height, largeness, speed, lifespan and gender.

Each is implemented through dedicated classes with internal logic for aging, reproduction, and interaction. It also models bodies and decomposition.

## Simulation
The `simulation` module adds the features needed to run the ecosystem:
- Random generation of organisms
- Various simulation modes (introducing organisms all at once or in stages)
- Data visualization
- Options for saving and restoring simulations

## Images
The images below are examples of the results.

## To-Do List
1. Give temperature a more significant role, possibly linked to metabolism (for example, ideal temperature increases energy gain).
2. Add more influential stats: animal behavior (solitary vs. group), senses (to affect movement logic), etc.
3. Increase complexity in plant reproduction (for example, interactions with nearby plants).
