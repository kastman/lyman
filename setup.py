#! /usr/bin/env python
#
# Copyright (C) 2012-2017 Michael Waskom <mwaskom@nyu.edu>

descr = """Lyman: A reproducible ecosystem for analyzing neuroimaging data."""

import os
from setuptools import setup

DISTNAME = 'lyman'
DESCRIPTION = descr
MAINTAINER = 'Michael Waskom'
MAINTAINER_EMAIL = 'mwaskom@nyu.edu'
LICENSE = 'BSD (3-clause)'
URL = 'http://www.cns.nyu.edu/~mwaskom/software/lyman/'
DOWNLOAD_URL = 'https://github.com/mwaskom/lyman'
VERSION = '2.0.0.dev'

def check_dependencies():

    # Just make sure dependencies exist, I haven't rigorously
    # tested what the minimal versions that will work are
    # TODO just do install_requires
    needed_deps = ["numpy", "pandas", "matplotlib", "scipy",
                   "nibabel", "nipype"]
    missing_deps = []
    for dep in needed_deps:
        try:
            __import__(dep)
        except ImportError:
            missing_deps.append(dep)

    if missing_deps:
        missing = ", ".join(missing_deps)
        raise ImportError("Missing dependencies: %s" % missing)

if __name__ == "__main__":

    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    import sys
    if not (len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or
            sys.argv[1] in ('--help-commands',
                            '--version',
                            'egg_info',
                            'clean'))):
        check_dependencies()

    setup(name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        version=VERSION,
        url=URL,
        download_url=DOWNLOAD_URL,
        install_requires=[],
        packages=['lyman',
                  'lyman.tests',
                  'lyman.workflows',
                  'lyman.workflows.tests'],
        scripts=['scripts/lyman'],
        classifiers=[
                     'Intended Audience :: Science/Research',
                     'Programming Language :: Python :: 2.7',
                     'Programming Language :: Python :: 3.6',
                     'License :: OSI Approved :: BSD License',
                     'Operating System :: POSIX',
                     'Operating System :: Unix',
                     'Operating System :: MacOS'],
    )
