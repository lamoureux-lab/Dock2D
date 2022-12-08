Protein
=======

Generate either a convex or concave hull based on specified `alpha` and `num_points`.
Hulls are created by radially distributing random points, then optimizing a perimeter based on the specified `alpha`.
Shapes can be convex by setting relatively lower `alpha` and higher `num_points`.
Hull coordinates are then converted to grid based shapes,
where pixels within the hull perimeter are assigned a value of 1, and 0 otherwise.

.. autoclass:: Dock2D.DatasetGeneration.Protein.Protein
   :special-members: __init__
   :members: generateConcave
   :undoc-members:

.. autofunction:: Dock2D.DatasetGeneration.Protein.get_random_points
.. autofunction:: Dock2D.DatasetGeneration.Protein.hull2array
