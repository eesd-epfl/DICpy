from distutils.core import setup

setup(
  name = 'DICpy',         # How you named your package folder (MyLib)
  packages = ['DICpy'],   # Chose the same as "name"
  version = '0.0.2',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Digital Image Correlation with Python',   # Give a short description about your library
  author = 'Ketson R. M. dos Santos',                   # Type in your name
  author_email = 'ketson.santos@epfl.ch',      # Type in your E-Mail
  url = 'https://github.com/eesd-epfl/DICpy',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/eesd-epfl/DICpy/archive/refs/tags/v0.0.2.tar.gz',    # I explain this later on
  keywords = ['DIC', 'Damage', 'Images'],   # Keywords that define your package best
  install_requires=[
      'numpy', 
      'scipy', 
      'matplotlib', 
      'scikit-learn', 
      'scikit-image',
      'opencv-python'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Science/Research',      # Define that your audience are developers
    'Topic :: Scientific/Engineering :: Image Processing',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
