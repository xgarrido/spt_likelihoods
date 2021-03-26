===============
SPT Likelihoods
===============

External likelihoods for SPT experiment using `cobaya <https://github.com/CobayaSampler/cobaya>`_.

.. image:: https://img.shields.io/github/workflow/status/xgarrido/spt_likelihoods/Unit%20test/feature-github-actions
   :target: https://github.com/xgarrido/spt_likelihoods/actions


Installing the code
-------------------

You can install the following code by just typing

.. code:: shell

    $ pip install https://github.com/xgarrido/spt_likelihoods.git


If you plan to develop/modify the code, the easiest way is to clone this repository to some location

.. code:: shell

    $ git clone https://github.com/xgarrido/spt_likelihoods.git /where/to/clone

Then you can install the likelihoods and its dependencies *via*

.. code:: shell

    $ pip install -e /where/to/clone

The ``-e`` option allow the developer to make changes within the ``spt`` directory without having
to reinstall at every changes. If you plan to just use the likelihood and do not develop it, you can
remove the ``-e`` option.

Installing SPT likelihood data
------------------------------

SPT data are stored in `LAMBDA <https://lambda.gsfc.nasa.gov/product/spt>`_. You can download them
by yourself but you can also use the ``cobaya-install`` binary and let it do the installation
job. For instance, if you do the next command

.. code:: shell

    $ cobaya-install /where/to/clone/examples/spt_example.yaml -p /where/to/put/packages

data and code such as `CAMB <https://github.com/cmbant/CAMB>`_ will be downloaded and installed
within the ``/where/to/put/packages`` directory. For more details, you can have a look to ``cobaya``
`documentation <https://cobaya.readthedocs.io/en/latest/installation_cosmo.html>`_.

Running/testing the code
------------------------

You can test the ``SPT`` likelihoods by doing

.. code:: shell

    $ python -m unittest spt.tests.test_spt_likelihoods
