from collections import namedtuple
import csv


def Area(a, b, c, d):
	altura = d - b
	largura = c - a
	return altura * largura


print(Area(10, 15, 25, 20))