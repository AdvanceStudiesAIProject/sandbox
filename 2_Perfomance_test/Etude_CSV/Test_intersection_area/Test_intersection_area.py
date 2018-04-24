from collections import namedtuple
Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

ra = Rectangle(3., 3., 5., 5.)
rb = Rectangle(1., 1., 10., 10.)
# intersection here is (3, 3, 4, 3.5), or an area of 1*.5=.5

def area(a, b):  # returns None if rectangles don't intersect
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    if (dx>=0) and (dy>=0):
        return dx*dy

print(area(ra, rb))