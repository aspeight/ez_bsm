import pytest

from itertools import product

import numpy as np
import pandas as pd

from ez_bsm.index_options import greeks_from

import ez_bsm.asay as fiv

def test_greeks_from_trivial():
    resp = greeks_from(requests=[], 
                    underlying=100., 
                    strike=100., 
                    horizon=1., 
                    volatility=0.2, 
                    int_rate=None, 
                    div_yld=None)
    assert len(resp) == 0

def test_greeks_from_all():
    resp = greeks_from(requests='all', 
                    underlying=100., 
                    strike=95., 
                    horizon=0.5, 
                    volatility=0.2, 
                    int_rate=0.02, 
                    div_yld=0.01)
    #print(resp)
    exp_resp = {
    'speed': -0.0010850623760921942, 'vega': 25.148243181424686, 
    'ultima': -287.1893811528756, 'gamma': 0.02514824318142468, 
    'vanna': -0.5820975124637006, 'put_price': 3.158049815598183, 
    'call_delta': 0.6769875541745383, 'vomma': 19.294580878008468, 
    'call_price': 8.60456352869545, 'put_delta': -0.31802492501814406,    
    }
    for k,v in exp_resp.items():
        assert pytest.approx(v) == resp[k]

def test_implied_vol_small_pricing_error():
    '''Make sure implied vol reprices accurately 
    
    The range of parameters is representative of equity index options.
    '''
    greeks_from = fiv.greeks_from

    taus = [1./365., 1./12., 0.5, 1., 2.]
    is_calls = [True, False]
    sigmas = [0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1., 1.3, 1.8]
    # Number of standard deviations for strike away from ATM
    zs = np.array([-5,-4,-3,-2,-1,0,1,2,3,4,5.])

    fwd = 100.
    rows = []
    for i,(tau, sigma, is_call, z) in enumerate(product(taus, sigmas, is_calls, zs)):
        strike = fwd*np.exp(z*sigma*np.sqrt(tau) - 0.5*tau*sigma**2)
        rows.append([fwd,tau,sigma,is_call,strike,z])
    df = pd.DataFrame(rows, columns=['fwd','tau','sigma','is_call', 'strike', 'z'])

    greeks = pd.DataFrame(
        greeks_from(fwd=df.fwd, strike=df.strike, tau=df.tau, sigma=df.sigma, is_call=df.is_call)
    )

    starting_sigma = 2.
    iv =fiv.safe_compute_iv(tgt=greeks['price'], fwd=df.fwd, 
                            strike=df.strike, tau=df.tau, 
                            is_call=df.is_call,
                            initial_sigma=starting_sigma,
                            sigma_bounds=(0.01,2.))
    
    new_greeks = pd.DataFrame(
        greeks_from(fwd=df.fwd, strike=df.strike, tau=df.tau, sigma=iv, is_call=df.is_call))

    assert np.abs(new_greeks.price - greeks.price).max() < 1e-10
    assert (iv-df.sigma).max() < 0.0001
