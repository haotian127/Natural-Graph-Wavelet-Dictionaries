## description for mat files

1. `sunflower_barbara.mat` contains
  * "L": sparse weighted unnormalized Laplacian matrix of the sunflower graph (N = 400 nodes)
  * "W": sparse weighted adjacency matrix of the sunflower graph
  * "X": N x 2 matrix, whose i-th row represents the `node i`'s xy coordinate.
  * "f_eye": the sunflower barbara eye signal, obtained by overlapping the sunflower graph with the barbara's eye (bilinear interpolation)
  * "f_face": the sunflower barbara face signal, obtained by overlapping the sunflower graph with the barbara's face (bilinear interpolation)
  * "f_trouser": the sunflower barbara trouser signal, obtained by overlapping the sunflower graph with the barbara's trouser (bilinear interpolation)

2. `toronto.mat` contains
  * "L": sparse weighted unnormalized Laplacian matrix of the toronto graph (N = 2275 nodes)
  * "W": sparse weighted adjacency matrix of the toronto graph
  * "X": N x 2 matrix, whose i-th row represents the `node i`'s xy coordinate.
  * "f_pedestrian": real pedestrian counts between the hours of 7:30am and 6:00pm on a single day measured during the period 03/22/2004–02/28/2018
  * "f_vehicle": real vehicle counts between the hours of 7:30am and 6:00pm on a single day measured during the period 03/22/2004–02/28/2018
  * "f_density": the node density is a smooth synthetic data (i.e., open a circle with fixed radius at each node and count number of nodes within the circle)

3. `RGC100_thickness_signal.mat` contains
  * "f": dendritic branch neuron thickness data on the RGC100 tree graph (N = 1154 nodes)
  * "g": add 8db white noise to the original signal f.
