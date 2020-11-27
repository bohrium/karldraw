''' author: samuel tenka
    change: 2020-11-27 
    create: 2019-03-25 
    descrp: render SGD diagrams   
    to use:
'''

from skimage.draw import circle, circle_perimeter_aa, line_aa, rectangle
import sys
import matplotlib.pyplot as plt
import numpy as np

black = np.array([[(0.0, 0.0, 0.0)]])
gray  = np.array([[(0.8, 0.8, 0.8)]])
red   = np.array([[(0.8, 0.2, 0.0)]])
teal  = np.array([[(0.1, 0.6, 0.4)]])

def draw_disk(img, row, col, rad=7, color=black):
    ''' render an anti-aliased solid disk onto the image '''
    # boundary:
    rr, cc, val = circle_perimeter_aa(row, col, rad)
    img[rr, cc, :] = np.minimum(img[rr, cc, :], 1.0 - np.expand_dims(val, 1) * (1.0 - color)[0])
    # interior:
    rr, cc = circle(row, col, rad)
    img[rr, cc, :] = color

def draw_line(img, row_s, col_s, row_e, col_e, color=black):
    ''' render an anti-aliased line onto the image '''
    rr, cc, val = line_aa(row_s, col_s, row_e, col_e)
    img[rr, cc, :] = np.minimum(img[rr, cc, :], 1.0 - np.expand_dims(val, 1) * (1.0 - color)[0])

#=============================================================================

_, filename = sys.argv

height, width = 480, 640
img = np.ones((height, width, 3), dtype=np.float32)
draw_disk(img, 100, 200, rad=10)
draw_disk(img, 150, 360, rad=20)
draw_line(img, 100, 200, 150, 360)

plt.imsave(filename, img)

