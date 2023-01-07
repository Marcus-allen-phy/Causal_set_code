# Causal_sets

This folder contains all the code needed to generate a Causal sets across 2 main geometries each with two sub-types.
- Rindler Horizon: A horizon formed from a null geodesic, with a measurement geodesic perpendicular. Along the meeting of these, is the horizon. In 1+1D this is a point, while in 2+1D it is a line. The horizon is enclosed within a diamond, such that all boundaries are null.
 - Normal: In 2+1, forms a diamond faced prism
 - Wrapped: Same as the Normal in 2+1 but has the added boundary condition that the end of the prism is mapped onto the other end of the prism, forming a diamond shaped torus. 
- Dynamical Horizon: Formed from the interception of a past light cone horizon being measure using a future light cone. In 1+1D, this forms two distinct point horizons, while in 2+1D a circular horizon is formed. The horizon is enclosed within a diamond, such that all boundaries are null.
 - Shell: In 2+1 the complete diamond is created and used in calculations.
 - Donut: The region inside the circular horizon is reduced in an attempt to limit the relations to just be those near the horizon.

## File descriptions

- Based_simulations: Shared Simulation class
- dynamical_donut: Specific causal set generator to form the dynamical horizon in a donut shape 
- dynamical_shell: Specific causal set generator to form the dynamical horizon in a shell shape
- GPU: Code used to run on a GPU
- methods: Common processes not liked to a particular horizon or independent of specific Causal set methods
- parallel: Majority of the code for calculations. All code is parallelized so it runs more optimally.
- rindler: Specific causal set generator to form a Rindler horizon

