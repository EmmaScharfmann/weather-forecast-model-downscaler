"""Operational forecast conversion keeps accumulated rain in physical units."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from scripts.precipitation_live import interval_fields, validate_message
from downscale.precip_data import forecast_pack


def test_cumulative_forecast_depths_become_hourly_mm_with_endpoint_states():
    values=np.ones((2,2,3,4),dtype='float32') * 280
    values[:,0]=np.array([.006,.018])[:,None,None]
    fields=xr.DataArray(values,dims=('lead_time','variable','lat','lon'),coords={
        'lead_time':np.array([6,12],dtype='timedelta64[h]'),'variable':['tp','t2m'],
        'lat':np.arange(3.),'lon':np.arange(4.),'time':np.datetime64('2026-11-15T06')})
    result=forecast_pack(interval_fields(fields),fields.lat,fields.lon,accumulation='interval')
    np.testing.assert_allclose(result.predictors.sel(variable='tp').values[:,0,0],[1]*6+[2]*6,rtol=1e-6)
    np.testing.assert_allclose(result.predictors.sel(variable='t2m'),280)
    assert pd.Timestamp(result.time[0].values)==pd.Timestamp('2026-11-15T07')
    assert pd.Timestamp(result.time[-1].values)==pd.Timestamp('2026-11-15T18')
    broken=fields.copy();broken.loc[dict(variable='tp',lead_time=fields.lead_time[-1])]=0
    with pytest.raises(ValueError,match='resets'):
        interval_fields(broken)


def test_grib_validation_rejects_wrong_run_or_interval(monkeypatch):
    import eccodes
    metadata=dict(shortName='tp',units='kg m**-2',dataDate=20261115,dataTime=600,
                  endStep=12,startStep=0,stepType='accum')
    monkeypatch.setattr(eccodes,'codes_new_from_message',lambda message:1)
    monkeypatch.setattr(eccodes,'codes_get',lambda handle,key:metadata[key])
    monkeypatch.setattr(eccodes,'codes_release',lambda handle:None)
    assert validate_message(b'grib','tp','2026-11-15T06',12)=='kg m**-2'
    metadata['startStep']=6
    with pytest.raises(ValueError,match='startStep'):
        validate_message(b'grib','tp','2026-11-15T06',12)
    metadata['startStep']=0
    with pytest.raises(ValueError,match='dataTime'):
        validate_message(b'grib','tp','2026-11-15T00',12)


def test_quantized_accumulation_uses_monotone_envelope_without_extra_water():
    fields=xr.DataArray(np.array([1.,.95,2.],dtype='float32')[:,None,None,None]/1000,
                       dims=('lead_time','variable','lat','lon'),coords={'variable':['tp']})
    result=interval_fields(fields,np.array([.00005]*3))
    np.testing.assert_allclose(result.sel(variable='tp').values.ravel()*1000,[1,0,1],rtol=1e-6)
    assert result.attrs['max_cumulative_quantization_correction_mm']==pytest.approx(.05,abs=1e-6)
    with pytest.raises(ValueError,match='resets'):
        interval_fields(fields,np.array([.000001]*3))
