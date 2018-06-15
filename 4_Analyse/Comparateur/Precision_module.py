from collections import namedtuple

Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')

def Area(rect):
	if rect == None:
		return 0
	else:
		a, b, c, d = rect
		return c * d

def IntersecArea(recta, rectb):  # returns None if rectangles don't intersect  (enter the cornones of initial point and the height ang width)
	if recta == None:
		return 0
	elif rectb == None:
		return 0
	else:
		xa, ya, wa, ha = recta
		a = Rectangle(xa, ya, xa+wa, ya+ha)
		xb, yb, wb, hb = rectb
		b = Rectangle(xb, yb, xb+wb, yb+hb)
		dx = int(min(a.xmax, b.xmax)) - int(max(a.xmin, b.xmin))
		dy = int(min(a.ymax, b.ymax)) - int(max(a.ymin, b.ymin))
		if (dx>=0) and (dy>=0):
			return int(dx*dy)
		else:
			return 0

def UnionAreas(rect1, rect2):
	intersection = IntersecArea(rect1, rect2)
	if intersection == None:
		return Area(rect1) + Area(rect2)
	elif intersection == 0:
		return Area(rect1) + Area(rect2)
	else:	
		somareas = Area(rect1) + Area(rect2)
		return somareas - intersection

def Precision(rect_input, rect_GT):
	#Aways respct the orther of input on the function
	porcetages = []
	for i in range(0, len(rect_input)):
		for j in range(0, len(rect_GT)):
			if UnionAreas(rect_input[i], rect_GT[j]) == 0:
				return [100] #case where there is no detectin in both, Input and GT
			porcetages.append((IntersecArea(rect_input[i], rect_GT[j])/UnionAreas(rect_input[i], rect_GT[j]))*100)
	return porcetages