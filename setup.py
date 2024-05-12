from distutils.core import setup
from setuptools import setup, find_packages

setup(
  name = 'cloudrtr',         # How you named your package folder (MyLib)
  packages=find_packages('src'),
  package_dir={'': 'src'},
  version = '0.0.32',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'System of Systems Multi-Agent Architecture',   # Give a short description about your library
  author = 'SMERTZANI',                   # Type in your name
  author_email = 'mertzanisemina@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/amertzani/rsos2.git',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/amertzani/rsos2/archive/refs/tags/rsos2.tar.gz',    # I explain this later on
  keywords = ['Multi-agent', 'Reinforcement Learning'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'scipy','mesa','pandas','gym','matplotlib'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)