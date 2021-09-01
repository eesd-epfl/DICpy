.. _instroduction_doc:

Introduction
====================

Digital Image Correlation (DIC) is a class of non-contacting methods for extracting strain fields, deformation, and motion of objects by using digital image registration. It is relevant for solving problem in several fields such as in civil engineering [2]_, biomechanics biomechanics [3]_, and materials science [4]_. In DIC matching algorithms are used to assess the deformation of objects, however, in most applications there is a need for measure very small deformations and therefore, higher image resolutions are required. To further increase the precision of the measurements sub-pixel registration algorithms are employed for minimal systematic errors [5]_. In general, DIC can be classified as 2D and 3D [1]_. 2D DIC is focused on the estimation of strain fields in materials by comparing pictures at consecutive load steps. Therefore, matching algorithms are used for maximize the correlation of patches encoding the local deformation in two dimensions only. On the other hand, 3D DIC consider stereographic images acquired with multiple cameras for estimating the strain fields in three dimensions. Moreover, a similar approach can be applied for the analysis of volumetric images, in this regard digital volume correlation (DVC) can be applied [6]_.

In this regard, ``DICpy`` (Digital Image Correlation with Python) is developed as a library of algorithms used in DIC. By making them more accessible, users can easily use DIC on their applications. However, it is important mentioning that ``DICpy`` is in the experimental phase, and possible inconsistencies/bugs/errors must be taken into consideration by the users.

.. [1] H. Schreier, J-J Orteu, M. A. Sutton, Image Correlation for Shape, Motion and Deformation Measurements: Basic Concepts,Theory and Applications, ISBN: 978-0-387-78747-3, ed. 1, Springer, Boston, MA.

.. [2] A. Rezaie, R. Achanta, M. Godio, K. Beyer, Comparison of crack segmentation using digital image correlation measurements and deep learning, Construction and Building Materials, 261, 2020, pp. 120474.

.. [3] Y. Katz, Z. Yosibash, New insights on the proximal femur biomechanics using Digital Image Correlation, Journal of Biomechanics, 101, 2020, pp. 109599.

.. [4] S. R. Heinz, J. S. Wiggins, Uniaxial compression analysis of glassy polymer networks using digital image correlation, Polymer Testing, 29(8), 2010, pp. 925-932,

.. [5] B. Pan, B. Wang. Digital Image Correlation with Enhanced Accuracy and Efficiency: A Comparison of Two Subpixel Registration Algorithms. Experimental Mechanics, 56, 2016, pp. 1395–1409.

.. [6] B. K. Bay, T. S. Smith, D. P. Fyhrie et al. Digital volume correlation: Three-dimensional strain mapping using X-ray tomography. Experimental Mechanics 39, 1999, 217–226.

.. toctree::
    :maxdepth: 2










