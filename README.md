# control_theory_fly_pursuit
Rotation work for Chiappe Lab

Folder descriptions

gif_images: dump of images from animations that then together form a gif

figures: figures generated whilst learning about control theory and testing some controllers based on learning_PIDs.py

playground_figs: the bulk of the simulation results. Sub folders within for each virtual path type. Within those are figures of simulations with different params

real_flies: similar to playground_figs but applied to real experimentally-controlled paths instead of virtual paths

misc_docs: poster figures, final posters and other misc docs



File descriptions

learning_PIDs.py: first implementations of P,I,D architectures based on sources online

va_controller.py: implementation of angular velocity controller

va_vs_parallel.py: implementation of angular and sideways velocities in a parallel controller architecture

explore_real_data.py: first plots with data from experiments, also serves to extract fly data with highest perc of chase.
