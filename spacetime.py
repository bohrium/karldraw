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

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~  0. rendering primitives  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

white = np.array([[(1.0, 1.0, 1.0)]])
black = np.array([[(0.0, 0.0, 0.0)]])
gray  = np.array([[(0.8, 0.8, 0.8)]])
red   = np.array([[(0.8, 0.2, 0.0)]])
teal  = np.array([[(0.1, 0.6, 0.4)]])

def draw_line(img, start, end, color=black):
    ''' render an anti-aliased line onto the image '''
    (row_s, col_s), (row_e, col_e) = start, end
    rr, cc, val = line_aa(row_s, col_s, row_e, col_e)
    img[rr, cc, :] = np.minimum(img[rr, cc, :], 1.0 - np.expand_dims(val, 1) * (1.0 - color)[0])

def draw_disk(img, center, radius=7, color=black, fillcolor=None):
    ''' render an anti-aliased circle or disk onto the image.  by default, the
        circle is hollow.  set `fillcolor` for a solid disk.
    '''
    (row, col) = center
    # interior:
    if fillcolor is not None:
        rr, cc = circle(row, col, radius)
        img[rr, cc, :] = fillcolor
    # boundary:
    rr, cc, val = circle_perimeter_aa(row, col, radius)
    img[rr, cc, :] = np.minimum(img[rr, cc, :], 1.0 - np.expand_dims(val, 1) * (1.0 - color)[0])

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~  1. coordinate system  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def make_converter(height, width, xmin, xmax, ymin, ymax):
    def rowcol_from_xy(xy): 
        x, y = xy
        return (
            int((height-1) * (1.0 - (y-ymin)/float(ymax-ymin))), # row 
            int((width -1) *        (x-xmin)/float(xmax-xmin)) , # col 
        ) 
    return rowcol_from_xy 

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#~~~~~  2. example scene  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def draw_example_scene():
    ''' draw a small kadinsky '''

    #---------  2.0. geometry  ------------------------------------------------
            
    height, width = 480, 480
    xmin, xmax, ymin, ymax = -1.0, 1.0, -1.0, 1.0
    img = np.ones((height, width, 3), dtype=np.float32)
    embed = make_converter(height, width, xmin, xmax, ymin, ymax)
    assert(float(height)/width==float(ymax-ymin)/(xmax-xmin))

    vv = 0.2 * np.array([1.0, 0.0])
    ww = 0.2 * np.array([(1.0+np.sqrt(5))/2, 1.0])
    oo = np.array([0.7, 0.0])
    m = np.array([[+1,0], [-0.1, +1]]) 

    #---------  2.1. drawing  -------------------------------------------------

    for o, v, w in ((0*oo, vv, ww), (-oo, np.matmul(m,vv), np.matmul(m,ww))):
        a = embed(o + 0*v + 0*w) 
        b = embed(o + 1*v + 0*w)
        c = embed(o + 2*v + 0*w)
        d = embed(o + 2*v + 1*w)
        e = embed(o + 1*v + 1*w)
        f = embed(o + 0*v + 1*w)

        # note that we draw lines before disks.  this way, the disk interiors will
        # paint over the lines' tips
        for p,q in ((a,b), (b,c), (c,d), (d,e), (e, f), (f,a)):
            draw_line(img, p, q)

        draw_disk(img, a, radius=4, fillcolor=black)
        draw_disk(img, b, radius=6, fillcolor=white)
        draw_disk(img, c, radius=4, fillcolor=black)
        draw_disk(img, d, radius=4, fillcolor=white)
        draw_disk(img, e, radius=6, fillcolor=black)
        draw_disk(img, f, radius=4, fillcolor=white)

    return img
        
if __name__ == '__main__':
    try:
        _, filename = sys.argv
    except ValueError:
        print('please enter one argument: an image filename to write to')
        exit()
    img = draw_example_scene()
    plt.imsave(filename, img)

