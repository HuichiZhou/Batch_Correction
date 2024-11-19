img_size = (512, 512) 
patch_size = (16, 16)

grid_size = tuple([s // p for s, p in zip(img_size, patch_size)]) # (512, 512) (16, 16)
