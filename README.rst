===============
SPT Likelihoods
===============

External likelihoods for SPT experiment using `cobaya
<https://github.com/CobayaSampler/cobaya>`_. These are ``python`` implementation of original ``Fortran``
code for ``CosmoMC`` sampler.

.. image:: https://img.shields.io/github/actions/workflow/status/xgarrido/spt_likelihoods/testing.yml
   :alt: GitHub Workflow Status
   :target: https://github.com/xgarrido/spt_likelihoods/actions

The package includes the following likelihoods:

- ``sptpol_2017`` relates to SPTPol EETE likelihood used in `Henning et al. <https://arxiv.org/abs/1707.09353>`_, 2017. The original ``Fortran`` code is available `here <https://pole.uchicago.edu/public/data/henning17/>`_ or in `LAMBDA <https://lambda.gsfc.nasa.gov/product/spt/sptpol_lh_2017_get.cfm>`_.

- ``spt3g_2020`` relates to SPT3G EETE likelihood used in `Dutcher et al. <https://arxiv.org/abs/2101.01684>`_, 2021. The original ``Fortran`` code is available `here <https://pole.uchicago.edu/public/data/dutcher21/#Likelihood>`_.

.. image:: https://user-images.githubusercontent.com/2495611/212894449-75255465-f4f6-42e5-a8b6-b4b03bb246d6.png

- ``spt_hiell_2020`` relates to SPT-SZ TT likelihood used in `Reichardt et al. <https://arxiv.org/abs/2002.06197>`_, 2020. The original ``Fortran`` code is available in `LAMBDA <https://lambda.gsfc.nasa.gov/product/spt/spt_ps_2020_get.cfm>`_.

.. image:: https://user-images.githubusercontent.com/2495611/212894382-d114c2ca-2659-47ee-ab56-ce66f8a81b1f.png

- ``spt3g_2022`` relates to SPT3G TT, TE and EE likelihood used in `Balkenhol et al. <https://arxiv.org/abs/2212.05642>`_, 2022. The original ``Fortran`` code is available `here <https://pole.uchicago.edu/public/data/balkenhol22/#Likelihood>`_.

.. image:: https://user-images.githubusercontent.com/2495611/214127005-aa514efa-4b73-43f7-a5ad-8274ffe151d9.png

Installing the code
-------------------

You can install the following code by just typing

.. code:: shell

    $ pip install git+https://github.com/xgarrido/spt_likelihoods.git


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

    $ cobaya-install /where/to/clone/examples/spt3g_example.yaml -p /where/to/put/packages

data for SPT3G and code such as `CAMB <https://github.com/cmbant/CAMB>`_ will be downloaded and
installed within the ``/where/to/put/packages`` directory. For more details, you can have a look to
``cobaya`` `documentation <https://cobaya.readthedocs.io/en/latest/installation_cosmo.html>`_.

Running/testing the code
------------------------

You can test the ``SPT`` likelihoods by doing

.. code:: shell

    $ test-spt

It will perform basic Χ² checks over the three different likelihoods.

You can also run MCMC chains with

.. code:: shell

    $ cobaya-run /where/to/clone/examples/spt3g_example.yaml -p /where/to/put/packages
