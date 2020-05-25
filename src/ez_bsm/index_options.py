'''Routines to compute prices and greeks of index options under the Black-Scholes-Merton model'''

import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats

_gaussian = scipy.stats.norm()

# For definitions, see
# https://en.wikipedia.org/wiki/Greeks_(finance)
# Note: theta is slightly different for calls and puts, 
# but the same if for zero int_rate and div_yield case.
# I don't typically use theta in practice.

_SUPPORTED_REQUESTS = {
    'call_price', 'put_price',
    'call_delta', 'put_delta',
    'gamma', 'vega',
    'vanna', 'vomma', 
    'speed', 'ultima',
}

_STANDARD_REQUESTS = {
    'call_price', 'put_price',
    'call_delta', 'put_delta', 
    'gamma', 'vega', 
}

def _impl_greeks_from( 
                spot, 
                strike, 
                tau, 
                sigma, 
                int_rate, 
                div_yld):
    '''Low-level impl of greeks_from''' 
    sqrt_tau = np.sqrt(tau)
    discount_factor = np.exp(-int_rate*tau)
    yield_factor = np.exp(-div_yld*tau)
    fwd = yield_factor * spot
    
    carry = int_rate - div_yld
    d0 = ((np.log(spot/strike) + carry*tau) / (sigma*sqrt_tau))
    d1 = ((np.log(spot/strike) + carry*tau + 0.5*sigma**2 * tau) 
          / (sigma*sqrt_tau))
    d2 = d1 - sqrt_tau * sigma
    
    Nd1 = _gaussian.cdf(d1)
    phid1 = _gaussian.pdf(d1)
    Nd2 = _gaussian.cdf(d2)
    Nmd1 = _gaussian.cdf(-d1)
    Nmd2 = _gaussian.cdf(-d2)
    
    call_price = fwd * Nd1 - discount_factor * strike * Nd2
    put_price = -fwd * Nmd1 + discount_factor * strike * Nmd2
    
    call_delta = yield_factor * Nd1
    put_delta = -yield_factor * Nmd1
    
    # Note: this calculation is not accuruate for extreme parameter values
    # May need to separately compute call and put versions?
    theta = ((-fwd*sigma*phid1) / (2*sqrt_tau) 
             - int_rate * strike *discount_factor * Nd2
             + div_yld*fwd*Nd1)
    gamma = (yield_factor * phid1) / (spot * sigma * sqrt_tau)
    speed = -(gamma / spot) * (1. + d1 / (sigma * sqrt_tau))
    vega = fwd * sqrt_tau * phid1
    vanna = (vega/spot)*(1-d1/(sigma*sqrt_tau))
    vomma = vega * d1 * d2 / sigma
    ultima = (-vega/sigma**2)*(d1*d2*(1-d1*d2)+d1**2+d2**2)
    
    return (call_price, put_price,
            call_delta, put_delta,
            gamma, speed,
            vega, vanna, vomma, ultima)


def greeks_from(requests, 
                underlying, 
                strike, 
                horizon, 
                volatility, 
                int_rate=None, 
                div_yld=None):
    '''Compute requested prices and greeks
    
    Arguments can be scalars, numpy arrays or pandas Series
    '''
    if requests is None:
        requests = _STANDARD_REQUESTS
    elif isinstance(requests, str) and requests.lower() == 'all':
        requests = _SUPPORTED_REQUESTS
    else:
        requests = set(requests)
        assert requests.issubset(_SUPPORTED_REQUESTS)

    if int_rate is None:
        int_rate = 0.
    if div_yld is None:
        div_yld = 0.

    (call_price, put_price,
     call_delta, put_delta,
     gamma, speed,
     vega, vanna, vomma, ultima) = _impl_greeks_from(spot=underlying, 
                                                     strike=strike,
                                                     tau=horizon,
                                                     sigma=volatility,
                                                     int_rate=int_rate,
                                                     div_yld=div_yld)
    
    raw_result = dict(call_price=call_price, put_price=put_price, 
                      call_delta=call_delta, put_delta=put_delta,
                      gamma=gamma, speed=speed,
                      vega=vega, vanna=vanna, vomma=vomma, ultima=ultima,
    )

    result = {k:raw_result[k] for k in requests}

    return result
