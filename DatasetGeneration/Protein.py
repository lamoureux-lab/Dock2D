import numpy as np
import seaborn as sea
sea.set_style("whitegrid")

from random import uniform
import alphashape
import shapely.geometry as geom


def get_random_points(num_points, xspan, yspan, debug=False):
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

	if debug:
		unzipped_points = list(zip(*points))
		plt.suptitle('Number of points distributed')
		plt.title('Input='+str(num_points)+', Output='+str(len(points)))
		plt.scatter(unzipped_points[0], unzipped_points[1])
		plt.show()

	return points


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
	def generateConcave(cls, size=50, alpha=1.0, num_points=25, occupancy=0.8, debug=False):
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
		points = get_random_points(num_points, points_coordinate_span, points_coordinate_span, debug)
		optimal = alpha * alphashape.optimizealpha(points, silent=False)
		hull = alphashape.alphashape(points, optimal)
		bulk = hull2array(hull, np.zeros((size, size)), grid_coordinate_span, grid_coordinate_span)
		return cls(bulk, hull=hull)


if __name__ == "__main__":
	import matplotlib.pyplot as plt
	prot = Protein.generateConcave(alpha=0.8, num_points=100, debug=True)

	def plot_coords(ax, poly, plot_alpha=0.25):
		x, y = poly.exterior.xy
		ax.fill(x, y, alpha=plot_alpha)

	fig, ax = plt.subplots()
	plot_coords(ax, prot.hull)
	plt.title('Hull generated')
	plt.show()

	plt.close()
	plt.imshow(prot.bulk.transpose())
	plt.title('Hull to array bulk')
	plt.gca().invert_yaxis()
	plt.show()
