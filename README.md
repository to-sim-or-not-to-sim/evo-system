# EVO-SYSTEM
A Python simulation of an evolving ecosystem using Perlin noise for procedural world generation.

## World Generation
The world is generated using Perlin noise, implemented through the custom `noise` library. This library includes functions to create various environmental maps (height, temperature, humidity, etc.). A Gaussian mask can be applied to create an island-like shape in the center of the world. It also provides tools to visualize these maps. An example usage is included in the `main()` function.

## Organisms: Stats and Behavior
Organisms are defined in the custom `organism` library, which includes all the constants related to their needs, reproduction chances, and energy gain.

- **Plant** stats: height, roots, leaves and lifespan.
- **Animal** stats: height, largeness, speed, lifespan and gender.

Each is implemented through dedicated classes with internal logic for aging, reproduction, and interaction. It also models bodies and decomposition.

## Simulation
The `simulation` module adds the features needed to run the ecosystem:
- Random generation of organisms
- Various simulation modes (introducing organisms all at once or in stages)
- Data visualization
- Options for saving and restoring simulations

## Images
The images below are examples of simulation results:
1. [animation of organisms population (converted into .mp4)](biomap.mp4)
2. [plot of all organisms population](0_organisms_number.png)
3. [plot of herbivores population](0_herbivores_number.png)
4. [plot of carnivores population](0_carnivores_number.png)
5. [plot of plants stats](0_plants_stats.png)
6. [plot of herbivores stats](0_herbivores_stats.png)
7. [plot of carnivores stats](0_carnivores_stats.png)
8. [map](map.png)
9. [land and water divided](land.png)
10. [plot of number of herbivores and carnivores](0_herbivores-carnivores.png)
11. [animation of plot of organisms population](organisms.gif)

## To-Do List
1. Give temperature a more significant role, possibly linked to metabolism (for example, ideal temperature increases energy gain).
2. Add more influential stats: animal behavior (solitary vs. group), senses (to affect movement logic), etc.
3. Increase complexity in plant reproduction (for example, interactions with nearby plants).
4. Add underwater and amphibious life.
5. Add omnivorous animals.
6. Optimization and code cleanup.

---

**Note 1**: This is a personal, work-in-progress project built for fun and out of interest in evolving ecosystems. There is no guarantee of continued development, but I wanted to share this project because I think it's a cool and inspiring experiment.   
**Note 2**: If you have ideas or suggestions, feel free to reach out or open an issue!
