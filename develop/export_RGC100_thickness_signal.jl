using MAT

vars = matread("datasets/Dendrite.mat")

G_3D = vars["G_3D"]
f = G_3D["f"][:]
file = matopen("datasets/RGC100_thickness_signal.mat", "w")
write(file, "f", f)
write(file, "g", g)
close(file)

matread("datasets/RGC100_thickness_signal.mat")
