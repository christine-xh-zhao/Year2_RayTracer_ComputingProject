# Physics Year 2 Computing Project Ray Tracer - Imperial College London

## About the code

### Ray-tracer main module - the code does all the work

#### General functions

From `ray.py`

- Generate rays, ray bundles, a list of ray bundles

From `optical_element.py`

- Generate optical elements including refracting surfaces (spherical or plano) and output plane

- Propagate rays through the optical elements

- Calculate the focal length of any lens system

From `propagate_and_plot.py`

- Plot ray trajectories in 3D and 2D, and plot incident and output spot diagrams in 2D


#### For a plano-convex lens

From `propagate_and_plot.py`

- Plot the RMS spot radius (a measure of spherical aberration) against ray bundle radius for both orientations

#### For a biconvex lens

From `biconvex_optimization.py`

- Optimise the biconvex lens for its surface curvatures and/or the distance between its two surfaces

- Plot the aberration scale against the ray bundle radius for both the diffraction scale and the RMS spot radius of a given lens


### Plots 

#### Requested in tasks

To generate the plots, use `task_plots.py`

All plots generated can be found in `Plots_from_tasks` folder

#### Plots for testing

To generate the plots, use `test.py`

All plots generated can be found in `Plots_from_tests` folder


### Testing

All in `test.py`


## Summary report

Can be found in `Report` folder

Final mark of the entire project: 21/25

Hope this repo can help you :)
