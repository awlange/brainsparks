for i in range(0, image_width-kernel_width+1):
    for j in range(0, image_height-kernel_height+1):
        for x in range(0, kernel_width):
            for y in range(0, kernel_height):
                sum += kernel[x,y] * image[i+x,j+y]

        feature_map[i,j] = act_func(sum)
        sum = 0.0