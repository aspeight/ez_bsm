import pytest

from itertools import product

import numpy as np
import pandas as pd

from ez_bsm.index_options import greeks_from

import ez_bsm.asay as fiv

def test_greeks_from_accuracy():
    '''Make sure greeks are reasonable based on finite difference approximations'''
    is_calls = [True,False]
    strikes = [85., 90., 100., 110, 120.]

    for is_call,strike in product(is_calls,strikes):
        params = dict(strike=strike, is_call=is_call, int_rate=0.05, div_yld=0.02)

        ds = 0.001
        s0 = 100.
        dv = 0.00001
        v0 = 0.25
        spots = np.array([s0-3*ds, s0-2*ds, s0-ds, s0, s0+ds, s0+2*ds, s0+3*ds])
        vols = np.array([v0-3*dv, v0-2*dv, v0-dv, v0, v0+dv, v0+2*dv, v0+3*dv])
        df = pd.DataFrame(product(spots,vols), columns=['spot','sigma'])

        tau0 = 0.25
        dt = 0.00001

        greeks = greeks_from(spot=df.spot, sigma=df.sigma, tau=tau0, **params)
        gr0 = greeks_from(spot=s0, sigma=v0, tau=tau0, **params)
        gr1 = greeks_from(spot=s0, sigma=v0, tau=tau0+dt, **params)
        grm1 = greeks_from(spot=s0, sigma=v0, tau=tau0-dt, **params)

        df['price'] = greeks['price']

        price = df.pivot('spot','sigma','price')

        fds = {'price':price}

        fds['delta'] = (price.shift(-1) - price.shift(1)) / (2*ds)
        fds['gamma'] = (fds['delta'].shift(-1) - fds['delta'].shift(1)) / (2*ds)
        fds['speed'] = (fds['gamma'].shift(-1) - fds['gamma'].shift(1)) / (2*ds)
        fds['vega'] = (price.shift(-1,axis=1) - price.shift(1,axis=1)) / (2*dv)
        fds['vanna'] = (fds['vega'].shift(-1) - fds['vega'].shift(1)) / (2*ds)
        fds['vomma'] = (fds['vega'].shift(-1,axis=1) - fds['vega'].shift(1,axis=1)) / (2*dv)
        fds['theta'] = -(gr1['price'] - grm1['price'])/(2*dt)

        for k,v in fds.items():
            vv = v if k=='theta' else v.iloc[3,3]
            assert np.abs(gr0[k]-vv) < 1e-4


def test_greeks_from():
    resp = greeks_from(spot=100., 
                    strike=95., 
                    tau=0.5, 
                    sigma=0.2, 
                    int_rate=0.02, 
                    div_yld=0.01,
                    is_call=True)
    #print(resp)

    exp_resp = {'call_price': 8.60456352869545, 'put_price': 3.158049815598183, 
            'call_delta': 0.6769875541745383, 'put_delta': -0.31802492501814406, 
            'call_theta': -5.534544919885566, 'put_theta': -4.64846271495483, 
            'gamma': 0.02514824318142468, 'speed': -0.0010850623760921942, 'vega': 25.148243181424686, 
            'vanna': -0.5820975124637006, 'vomma': 19.294580878008468, 'ultima': -287.1893811528756, 
            'price': 8.60456352869545, 'delta': 0.6769875541745383, 'theta': -5.534544919885566} 

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

