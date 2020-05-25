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
    phimd1 = _gaussian.pdf(-d1)
    
    call_price = fwd * Nd1 - discount_factor * strike * Nd2
    put_price = -fwd * Nmd1 + discount_factor * strike * Nmd2
    
    call_delta = yield_factor * Nd1
    put_delta = -yield_factor * Nmd1
    

    call_theta = ((-fwd*sigma*phid1) / (2*sqrt_tau) 
             - int_rate * strike *discount_factor * Nd2
             + div_yld*fwd*Nd1)
    put_theta = ((-fwd*sigma*phimd1) / (2*sqrt_tau) 
             + int_rate * strike *discount_factor * Nmd2
             - div_yld*fwd*Nmd1)
    gamma = (yield_factor * phid1) / (spot * sigma * sqrt_tau)
    speed = -(gamma / spot) * (1. + d1 / (sigma * sqrt_tau))
    vega = fwd * sqrt_tau * phid1
    vanna = (vega/spot)*(1-d1/(sigma*sqrt_tau))
    vomma = vega * d1 * d2 / sigma
    ultima = (-vega/sigma**2)*(d1*d2*(1-d1*d2)+d1**2+d2**2)
    
    return (call_price, put_price,
            call_delta, put_delta,
            call_theta, put_theta,
            gamma, speed,
            vega, vanna, vomma, ultima)


def greeks_from(spot, 
                strike, 
                tau, 
                sigma, 
                is_call=None,
                int_rate=None, 
                div_yld=None):
    '''Computes prices and greeks for European options in the Black Scholes model'''

    if int_rate is None:
        int_rate = 0.
    if div_yld is None:
        div_yld = 0.

    (call_price, put_price,
     call_delta, put_delta,
     call_theta, put_theta,
     gamma, speed,
     vega, vanna, vomma, ultima) = _impl_greeks_from(spot=spot, 
                                                     strike=strike,
                                                     tau=tau,
                                                     sigma=sigma,
                                                     int_rate=int_rate,
                                                     div_yld=div_yld)
    
    result = dict(    call_price=call_price, put_price=put_price, 
                      call_delta=call_delta, put_delta=put_delta,
                      call_theta=call_theta, put_theta=put_theta,
                      gamma=gamma, speed=speed,
                      vega=vega, vanna=vanna, vomma=vomma, ultima=ultima,
        )

    if is_call is not None:

        result['price'] = is_call * call_price + (1-is_call)*put_price
        result['delta'] = is_call * call_delta + (1-is_call)*put_delta
        result['theta'] = is_call * call_theta + (1-is_call)*put_theta


    return result
