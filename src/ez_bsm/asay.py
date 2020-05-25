'''Black Scholes with in the Asay (1982) flavor.

This works in the "fully margined options on futures" setup.
Interest rates and dividend yields are taken to be zero.
All other parameters must be reinterpreted in this light.

This module contains calculators for prices, greeks and
implied volatility that are meant to operate on vectors
or dataframes efficiently.  As such, the methods are simple 
and do not attempt to isolate corner cases.  Rather, this
module seeks to provide acceptable performance on medium-to-large 
datasets with parameters that fall in the typical range of 
equity index options or options on futures.  For example,
Options with deltas far outside the range of [5,95] might
not be appropriate.  Nor would maturities outside the range
[1 week, 2 years] or implied vols outside [5%, 150%].

The paper by Li (2006) provides a useful domain for 
of reliability for iterative implied volatility calculations.

'''

__all__ = ['greeks_from', 'safe_compute_iv']

import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats

_gaussian = scipy.stats.norm()
_N_cdf = _gaussian.cdf
_N_pdf = _gaussian.pdf

def greeks_from(fwd, strike, tau, sigma, is_call):
    '''Computes Black-Scholes prices and greeks assuming zero interest and dividend.
    
    Inputs are assumed to be np.array or pandas.Series.

    Returns a dictionary 

    See 
    https://en.wikipedia.org/wiki/Greeks_(finance)
    '''
    sqrt_tau = np.sqrt(tau)
    log_fwd_over_strike = np.log(fwd/strike)
    d1 = (log_fwd_over_strike + 0.5*sigma**2 * tau) / (sigma*sqrt_tau)
    d2 = d1 - sqrt_tau * sigma
    
    Nd1 = _N_cdf(d1)
    phid1 = _N_pdf(d1)
    Nd2 = _N_cdf(d2)
    Nmd1 = _N_cdf(-d1)
    Nmd2 = _N_cdf(-d2)
    
    call_price = fwd * Nd1 - strike * Nd2
    put_price = -fwd * Nmd1 + strike * Nmd2
    
    call_delta = Nd1
    put_delta = -Nmd1
    
    # Note: in this special case, theta is the same for call and put options
    theta = -0.5*fwd * (sigma/sqrt_tau) * phid1 
    gamma = (phid1) / (fwd * sigma * sqrt_tau)
    speed = -(gamma / fwd) * (1. + d1 / (sigma * sqrt_tau))

    vega = fwd * sqrt_tau * phid1
    vanna = (vega/fwd)*(1-d1/(sigma*sqrt_tau))
    vomma = vega * d1 * d2 / sigma
    ultima = (-vega/sigma**2)*(d1*d2*(1-d1*d2)+d1**2+d2**2)
    
    result = dict(  call_price=call_price,
                    put_price=put_price,
                    call_delta=call_delta,
                    put_delta=put_delta,
                    theta=theta,
                    gamma=gamma,
                    speed=speed,
                    vega=vega,
                    vanna=vanna,
                    vomma=vomma,
                    ultima=ultima,
                )
    
    # todo: fill nan with zero before multiplying by bool
    result['price'] = 1.*is_call * call_price + (1.-is_call)*put_price
    result['delta'] = 1.*is_call * call_delta + (1.-is_call)*put_delta
    result['abs_delta'] = np.abs(result['delta'])

    return result

def refine_iv(tgt, price, vega, vomma, ultima, order=3):

    '''One iteration of Newton-like method for implied vol calculation

    A higher order generalization of Newton's method is supported.
    See https://en.wikipedia.org/wiki/Householder%27s_method

    Params
    ------
    tgt : (np.array) observed option price (calibration target)
    price : (np.array) model-computed price (call or put) given sigma
    vega, vomma, ultima : (np.array) model-computed greeks given sigma
    order : (int) 1=Newton's method, 2-3 are higher order Householder methods

    Returns
    -------
    An array (compatible with tgt) that, when added to the current
    implied volatility, gives an improved estimate. That is,
    iv -> iv + update.

    Notes
    -----
    The paper by Li (2006) provides a useful domain for when this
    type of iteration can be expected to converge:
    |x| <= 0.5, 0 << v <= 1, and |x/v| <= 2,
    where x = log(F/K), F=exp((r-q)*tau)*spot, and v = sqrt(tau)*sigma.

    Generally, starting with a sigma near the upper end of Li's domain
    gives good convergence rates.
    '''
    x = tgt - price
    h = x / vega
    if order==1:
        update = h
    elif order==2:
        update = h * (1./(1 + 0.5*(h)*(vomma/vega)))
    elif order==3:
        update = (h 
                  * (1 + 0.5*(vomma/vega)*h)
                  / (1 + (vomma/vega)*h + (1./6.)*(ultima/vega)*h**2 ))
    else:
        raise ValueError("order must be 1,2 or 3, not {}".format(order))
    return update

def raw_compute_iv(tgt, fwd, strike, tau, is_call,
                   initial_sigma=2., num_iters=12, order=3):
    '''Apply Newton-like iteration to solve for implied vol with no error checks.
    
    '''
    sigma = initial_sigma * (1. + 0*tgt)
    for it in range(num_iters):
        greeks = greeks_from(fwd=fwd, strike=strike, tau=tau, sigma=sigma, is_call=is_call)
        update = refine_iv(tgt=tgt, 
                           price=greeks['price'], 
                           vega=greeks['vega'], 
                           vomma=greeks['vomma'], 
                           ultima=greeks['ultima'], 
                           order=order)
        sigma += update
    return sigma

def safe_compute_iv(tgt, fwd, strike, tau, is_call,
                    initial_sigma=1.5,
                    num_iters=12, 
                    order=3,
                    sigma_bounds=(0.01,2.),
                    price_tol=None):
    '''Apply Newton-like iteration to solve for implied vol with some error checking
    
    '''
    greeks_low = greeks_from(fwd=fwd, strike=strike, tau=tau, sigma=sigma_bounds[0], is_call=is_call)
    greeks_high = greeks_from(fwd=fwd, strike=strike, tau=tau, sigma=sigma_bounds[1], is_call=is_call)

    clip_tgt = np.clip(tgt, greeks_low['price'], greeks_high['price'])
    
    iv = raw_compute_iv(clip_tgt, fwd, strike, tau, is_call,
                        initial_sigma=initial_sigma,
                        num_iters=num_iters, order=order)

    greeks = greeks_from(fwd=fwd, strike=strike, tau=tau, sigma=iv, is_call=is_call)
    #iv[clip_tgt != tgt] = np.nan # todo: float equality check is sometimes not what we want
    if price_tol is not None:
        iv[np.abs(greeks['price']-tgt)>price_tol] = np.nan
    return iv