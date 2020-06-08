# metric: HAD

G, L, X = threeClustersGraph()
gplot(1.0*adjacency_matrix(G), X; width = 1); scatter_gplot!(X; marker = IDX)
