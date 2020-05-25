======
ez_bsm
======
Calculate prices, greeks and implied volatilities for equity index options
under the Black-Scholes-Merton model.


Description
===========

The routines here are meant to be used in a vectorized manner. 
It plays nice with numpy and pandas and can process thousands
of option contracts at a time with good performance.  

.. code-block:: python

    contracts = pd.DataFrame(
        dict(strike=[90., 100., 110.],
            sigma=[0.3, 0.24, 0.21],
            is_call=[False, False, True]))

    df = pd.DataFrame(
        greeks_from(spot=100., 
                    strike=contracts.strike,
                    tau=0.25,
                    sigma=contracts.sigma,
                    is_call=contracts.is_call))



See the notebooks for examples on how to use this package.

This package is not meant to handle options that have deltas approaching
one or zero.  In particular, implied volatilitiy calculations may not
be robust in that case.  If you need that,  see "Let's be rational"
by Peter Jaeckel.

http://www.jaeckel.org/


Note
====

This project has been set up using PyScaffold 3.2.3. For details and usage
information on PyScaffold see https://pyscaffold.org/.
