Heroku buildpack: Python, Numpy, Scipy, Scikit-learn
====================================================

This is a [Heroku buildpack](http://devcenter.heroku.com/articles/buildpacks)
for Python apps, powered by [pip](http://www.pip-installer.org/).

Additionally, it adds support for Numpy, Scipy and Scikit-learn.

[![Build Status](https://secure.travis-ci.org/dbrgn/heroku-buildpack-python-sklearn.png?branch=master)](http://travis-ci.org/dbrgn/heroku-buildpack-python-sklearn)

Differences to other forks
--------------------------

This buildpack is strongly inspired by @wyn and @ToonTimbermont. In fact, I
copied a lot of code from them. It also uses binaries provided by @wyn. Thanks
a lot.

In contrast to their forks, this buildpack does not require a `setup.py`, it
works with a normal `requirements.txt` file. Additionally, it is based on the
current version of the heroku-buildpack-python. This means that it uses a
current version of pip, which gets rid of some stack traces in the deploy log.

Setup, Usage
------------

First of all, it is important that your requirements for Numpy and SciPy use
only one of the versions available as precompiled binaries here:
https://github.com/dbrgn/npscipy-binaries At the time of this writing,
supported versions are:

- Numpy 1.6.1
- Numpy 1.7.0
- Scipy 0.10.1
- Scipy 0.11.0

Then specify the buildpack as usual. For a new app:

    heroku create --buildpack https://github.com/dbrgn/heroku-buildpack-python-sklearn/

For an existing app:

    heroku config:set BUILDPACK_URL=https://github.com/dbrgn/heroku-buildpack-python-sklearn/

Demo
----

    $ mkdir testheroku
    $ cd testheroku
    $ git init
    $ heroku create --buildpack https://github.com/dbrgn/heroku-buildpack-python-sklearn/
    $ echo -e "numpy==1.7.0\nscipy==0.11.0\nscikit-learn==0.13.1" > requirements.txt
    $ git add requirements.txt
    $ git commit -m 'Added requirements'
    $ git push heroku master
    Counting objects: 3, done.
    Writing objects: 100% (3/3), 260 bytes | 0 bytes/s, done.
    Total 3 (delta 0), reused 0 (delta 0)

    -----> Fetching custom git buildpack... done
    -----> Python app detected
    -----> No runtime.txt provided; assuming python-2.7.4.
    -----> Preparing Python runtime (python-2.7.4)
    -----> Installing Distribute (0.6.36)
    -----> Installing Pip (1.3.1)
    -----> Noticed numpy/scipy. Bootstrapping prebuilt binaries.
    -----> Creating/downloading binaries.
    -----> Creating/downloading numpy bdist.
    -----> Creating/downloading scipy bdist.
    -----> Moving everything from venv directory to python directory...
    -----> Installing dependencies using Pip (1.3.1)
    -----> Noticed scikit-learn. Installing...
    -----> Setting environment vars...
    -----> Installing scikit-learn==0.13.1 via pip
    Downloading/unpacking scikit-learn==0.13.1
      Running setup.py egg_info for package scikit-learn

    (...)

    Successfully installed scikit-learn
    Cleaning up...
           Cleaning up...

    -----> Discovering process types
           Procfile declares types -> (none)

    -----> Compiled slug size: 59.6MB
    -----> Launching... done, v4
           http://ancient-eyrie-7305.herokuapp.com deployed to Heroku

    To git@heroku.com:ancient-eyrie-7305.git
     * [new branch]      master -> master
