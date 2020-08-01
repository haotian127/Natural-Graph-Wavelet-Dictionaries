using MAT

## sunflower barbara
sunflower_barbara_file = matopen("datasets/sunflower_barbara.mat", "w")
write(sunflower_barbara_file, "W", W)
write(sunflower_barbara_file, "L", L)
write(sunflower_barbara_file, "X", X)
write(sunflower_barbara_file, "f_eye", f_eye_Bilinear)
write(sunflower_barbara_file, "f_face", f_face_Bilinear)
write(sunflower_barbara_file, "f_trouser", f_trouser_Bilinear)
close(sunflower_barbara_file)

# double check the saved mat file works
vars = matread("datasets/sunflower_barbara.mat")


## toronto
toronto_file = matopen("datasets/toronto.mat", "w")
write(toronto_file, "W", Weight)
write(toronto_file, "L", sparse(L))
write(toronto_file, "X", X)
write(toronto_file, "f_pedestrian", fp)
write(toronto_file, "f_vehicle", fv)
write(toronto_file, "f_density", f)
close(toronto_file)

# double check the saved mat file works
vars = matread("datasets/toronto.mat")
