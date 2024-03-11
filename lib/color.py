RED = (234, 20, 28)
GREEN = (83, 218, 63)
BLUE = (0, 0, 255)
YELLOW = (254, 251, 52)
MAGENTA = (255, 0, 255)
CYAN = (1, 237, 250)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (50,50,50) #(128, 128, 128)
LIGHT_GRAY = (192, 192, 192)
DARK_GRAY = (0, 0, 0)   #(64, 64, 64)
PINK = (255, 175, 175)
ORANGE = (255, 145, 12)
PURPLE = (221, 10, 178)
BORDER = (32, 79, 145)

colors = [GRAY, RED, GREEN, CYAN, PURPLE, BLUE, ORANGE, YELLOW, PINK, (0, 237, 165), (22, 172, 237)]
cmap = dict()
for i in range(17):
    if i < len(colors):
        cmap[i] = colors[i]
    else:
        cmap[i] = ORANGE
