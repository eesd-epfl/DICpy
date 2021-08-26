.. _DIC2Local_doc:

DIC2D
====================

.. automodule:: DICpy.DIC2Local

This module contains the classes and methods to perform the 2D digital image correlation given images
Of the object under analysis at different instants.

The module ``DICpy.DIC2Local`` currently contains the following classes:

* ``RectangularMesh``: Class for the creation of mesh to perform local DIC.

* ``Analysis``: Class for performing the DIC.

* ``PostProcessing``: Class for the calculation and visualization of strain fields.


RectangularMesh
--------------------------------
	
Digital Image Correlation is a non-contact technique for measuring strain fields and deformations of materials using images. It is widely used in several fields such as civil engineering [1]_ and biomechanics [2]_.


The equation :math:`\mathcal{T}`, which is a flat inner-product space, is defined as a set of all tangent vectors at :math:`\mathcal{X}`; such as 

.. math:: \mathcal{T}_{\mathcal{X}}.


RectangularMesh Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``RectangularMesh`` class is imported using the following command:

>>> from DICpy.DIC2Local import RectangularMesh

One can use the following command to instantiate the class ``RectangularMesh``

.. autoclass:: DICpy.DIC2Local.RectangularMesh
    :members:  

Analysis
--------------------------------

The Class ``Analysis`` is used to run the analysis with pixel and sub-pixel resolution. 

Analysis Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``Analysis`` class is imported using the following command:

>>> from DICpy.DIC2Local import Analysis

One can use the following command to instantiate the class ``Analysis``

.. autoclass:: DICpy.DIC2Local.Analysis
    :members:  

PostProcessing
--------------------------------
	
The ``PostProcessing`` class is used to estimate and visualize the strain fields.

PostProcessing Class Descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``PostProcessing`` class is imported using the following command:

>>> from DICpy.DIC2Local import PostProcessing

One can use the following command to instantiate the class ``PostProcessing``

.. autoclass:: DICpy.DIC2Local.PostProcessing
    :members: 


|

.. [1] A. Rezaie, R. Achanta, M. Godio, K. Beyer, Comparison of crack segmentation using digital image correlation measurements and deep learning, Construction and Building Materials, 2020:p.120474.

.. [2] D. S. Zhang, D. D. Arola, Applications of digital image correlation to biological tissues, 2004, Journal of Biomedical Optics.

.. toctree::
    :maxdepth: 2

