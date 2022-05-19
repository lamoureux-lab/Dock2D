import numpy as np
import seaborn as sea
sea.set_style("whitegrid")

from random import uniform
import scipy
from scipy.spatial import ConvexHull
import alphashape
import shapely.geometry as geom


def get_random_points(num_points, xspan, yspan):
	"""
	Generate radially distributed random point coordinates.

	:param num_points: number of points used to generate hulls
	:param xspan: distance to span points in the `x` dimension
	:param yspan: distance to span points in the `y` dimension
	:return: points distribution
	"""
	points = [[uniform(*xspan), uniform(*yspan)]  for i in range(num_points)]
	radius_x = (xspan[1] - xspan[0])/2.0
	radius_y = (yspan[1] - yspan[0])/2.0
	center_x = (xspan[1] + xspan[0])/2.0
	center_y = (yspan[1] + yspan[0])/2.0
	points = [[x,y] for x,y in points if np.sqrt(((x-center_x)/radius_x) ** 2 + ((y-center_y)/radius_y) ** 2) < 1.0]
	points = np.array(points)
	return points


def is_p_inside_points_hull(hull, p):
	"""
	Check if convex hull points are within the hull or outside.

	:param hull: convex hull points
	:param p: current point
	:return: condition of current point `True` inside the hull, `False` otherwise.
	"""
	new_points = np.append(hull.points, p, axis=0)
	new_hull = ConvexHull(new_points)
	if list(hull.vertices) == list(new_hull.vertices):
		return True
	else:
		return False


def hull2array(hull, array, xspan, yspan):
	"""
	Convert a hull to a grid based shape. All grid pixels are filled with 1 inside the shape, 0 otherwise.

	:param hull: convex or concave hull points
	:param array: grid to convert hull to gridshape
	:param xspan: distance to span grid points in the `x` dimension
	:param yspan: distance to span grid points in the `y` dimension
	:return: bulk hull converted to a grid
	"""
	x_size = array.shape[0]
	y_size = array.shape[1]
	for i in range(x_size):
		for j in range(y_size):
			x = xspan[0] + i*float(xspan[1]-xspan[0])/float(x_size)
			y = yspan[0] + j*float(yspan[1]-yspan[0])/float(y_size)
			inside = False
			if isinstance(hull, ConvexHull):
				inside = is_p_inside_points_hull(hull, np.array([[x,y]]))
			elif isinstance(hull, geom.Polygon):
				inside = geom.Point(x,y).within(hull)
			if inside:
				array[i, j] = 1.0
			else:
				array[i, j] = 0.0
	return array


class Protein:

	def __init__(self, bulk, hull=None):
		"""
		:param bulk: protein shape bulk
		:param hull: coordinates for the convex/concave hull
		"""
		self.size = bulk.shape[0]
		self.bulk = bulk
		self.hull = hull

	@classmethod
	def generateConvex(cls, size=50, num_points=15, occupancy=0.8):
		"""
		Generate convex hull coordinates and convert to grid based shape
		filled with 1 inside the shape, 0 otherwise.

		:param size: the dimensions of the box containing the shape
		:param num_points: number of points randomly radially distributed to generate the hull
		:param occupancy: distance to span the the point distributions
		:return: convex protein shape bulk
		"""
		grid_coordinate_span = (0, size)
		points_coordinate_span = (0.5*(1.0-occupancy)*size, size - 0.5*(1.0-occupancy)*size)
		points = get_random_points(num_points, points_coordinate_span, points_coordinate_span)
		hull = scipy.spatial.ConvexHull(points)
		bulk = hull2array(hull, np.zeros((size, size)), grid_coordinate_span, grid_coordinate_span)
		return cls(bulk)

	@classmethod
	def generateConcave(cls, size=50, alpha=1.0, num_points=25, occupancy=0.8):
		"""
		Generate concave hull coordinates and convert to grid based shape
		filled with 1 inside the shape, 0 otherwise.

		:param size: the dimensions of the box containing the shape
		:param alpha: the level of concavity used to generate convex hulls
		:param num_points: number of points randomly radially distributed to generate the hull
		:param occupancy: distance to span the the point distributions
		:return: concave protein shape bulk
		"""
		grid_coordinate_span = (0, size)
		points_coordinate_span = (0.5*(1.0-occupancy)*size, size - 0.5*(1.0-occupancy)*size)
		points = get_random_points(num_points, points_coordinate_span, points_coordinate_span)
		optimal = alpha * alphashape.optimizealpha(points, silent=False)
		hull = alphashape.alphashape(points, optimal)
		bulk = hull2array(hull, np.zeros((size, size)), grid_coordinate_span, grid_coordinate_span)
		return cls(bulk, hull=hull)
