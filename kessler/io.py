# This code is part of Kessler, a machine learning library for spacecraft collision avoidance.
#
# Copyright (c) 2020-
# University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
# Trillium Technologies
# Giacomo Acciarini
# and other contributors, see README in root of repository.
#
# GNU General Public License version 3. See LICENSE in root of repository.


import pandas as pd

from . import util
from .cdm import CDM
from .event import Event, EventDataset


def from_pandas(df, cdm_compatible_fields={
    'relative_speed': 'RELATIVE_SPEED',
    'ccsds_cdm_vers': 'CCSDS_CDM_VERS',
    'creation_date': 'CREATION_DATE',
    'originator':'ORIGINATOR',
    'message_for':'MESSAGE_FOR',
    'message_id':'MESSAGE_ID',
    'tca':'TCA',
    'miss_distance':'MISS_DISTANCE',
    'relative_speed':'RELATIVE_SPEED',
    'relative_position_r':'RELATIVE_POSITION_R',
    'relative_position_t':'RELATIVE_POSITION_T',
    'relative_position_n':'RELATIVE_POSITION_N',
    'relative_velocity_r':'RELATIVE_VELOCITY_R',
    'relative_velocity_t':'RELATIVE_VELOCITY_T',
    'relative_velocity_n':'RELATIVE_VELOCITY_N',
    'start_screen_period':'START_SCREEN_PERIOD',
    'stop_screen_period':'STOP_SCREEN_PERIOD',
    'screen_volume_frame':'SCREEN_VOLUME_FRAME',
    'screen_volume_shape':'SCREEN_VOLUME_SHAPE',
    'screen_volume_x':'SCREEN_VOLUME_X',
    'screen_volume_y':'SCREEN_VOLUME_Y',
    'screen_volume_z':'SCREEN_VOLUME_Z',
    'screen_entry_time':'SCREEN_ENTRY_TIME',
    'screen_exit_time':'SCREEN_EXIT_TIME',
    'jspoc_probability':'COLLISION_PROBABILITY',
    't_object_designator':'OBJECT1_OBJECT_DESIGNATOR',
    't_catalog_name':'OBJECT1_CATALOG_NAME',
    't_object_name':'OBJECT1_OBJECT_NAME',
    't_international_designator':'OBJECT1_INTERNATIONAL_DESIGNATOR',
    't_object_type':'OBJECT1_OBJECT_TYPE',
    't_ephemeris_name':'OBJECT1_EPHEMERIS_NAME',
    't_covariance_method':'OBJECT1_COVARIANCE_METHOD',
    't_maneuverable':'OBJECT1_MANEUVERABLE',
    't_orbit_center':'OBJECT1_ORBIT_CENTER',
    't_ref_frame':'OBJECT1_REF_FRAME',
    't_gravity_model':'OBJECT1_GRAVITY_MODEL',
    't_atmospheric_model':'OBJECT1_ATMOSPHERIC_MODEL',
    't_n_body_perturbations':'OBJECT1_N_BODY_PERTURBATIONS',
    't_solar_rad_pressure':'OBJECT1_SOLAR_RAD_PRESSURE',
    't_earth_tides':'OBJECT1_EARTH_TIDES',
    't_intrack_thrust':'OBJECT1_INTRACK_THRUST',
    't_time_lastob_start':'OBJECT1_TIME_LASTOB_START',
    't_time_lastob_end':'OBJECT1_TIME_LASTOB_END',
    't_recommended_od_span':'OBJECT1_RECOMMENDED_OD_SPAN',
    't_actual_od_span':'OBJECT1_ACTUAL_OD_SPAN',
    't_obs_available':'OBJECT1_OBS_AVAILABLE',
    't_obs_used':'OBJECT1_OBS_USED',
    't_tracks_available':'OBJECT1_TRACKS_AVAILABLE',
    't_tracks_used':'OBJECT1_TRACKS_USED',
    't_residuals_accepted':'OBJECT1_RESIDUALS_ACCEPTED',
    't_weighted_rms':'OBJECT1_WEIGHTED_RMS',
    't_area_pc':'OBJECT1_AREA_PC',
    't_area_drg':'OBJECT1_AREA_DRG',
    't_area_srg':'OBJECT1_AREA_SRP',
    't_mass':'OBJECT1_MASS',
    't_cd_area_over_mass':'OBJECT1_CD_AREA_OVER_MASS',
    't_cr_area_over_mass':'OBJECT1_CR_AREA_OVER_MASS',
    't_thrust_acceleration':'OBJECT1_THRUST_ACCELERATION',
    't_sedr':'OBJECT1_SEDR',
    't_x':'OBJECT1_X',
    't_y':'OBJECT1_Y',
    't_z':'OBJECT1_Z',
    't_x_dot':'OBJECT1_X_DOT',
    't_y_dot':'OBJECT1_Y_DOT',
    't_z_dot':'OBJECT1_Z_DOT',
    't_cr_r':'OBJECT1_CR_R',
    't_ct_r':'OBJECT1_CT_R',
    't_ct_t':'OBJECT1_CT_T',
    't_cn_r':'OBJECT1_CN_R',
    't_cn_t':'OBJECT1_CN_T',
    't_cn_n':'OBJECT1_CN_N',
    't_crdot_r':'OBJECT1_CRDOT_R',
    't_crdot_t':'OBJECT1_CRDOT_T',
    't_crdot_n':'OBJECT1_CRDOT_N',
    't_crdot_rdot':'OBJECT1_CRDOT_RDOT',
    't_ctdot_r':'OBJECT1_CTDOT_R',
    't_ctdot_t':'OBJECT1_CTDOT_T',
    't_ctdot_n':'OBJECT1_CTDOT_N',
    't_ctdot_rdot':'OBJECT1_CTDOT_RDOT',
    't_ctdot_tdot':'OBJECT1_CTDOT_TDOT',
    't_cndot_r':'OBJECT1_CNDOT_R',
    't_cndot_t':'OBJECT1_CNDOT_T',
    't_cndot_n':'OBJECT1_CNDOT_N',
    't_cndot_rdot':'OBJECT1_CNDOT_RDOT',
    't_cndot_tdot':'OBJECT1_CNDOT_TDOT',
    't_cndot_ndot':'OBJECT1_CNDOT_NDOT',
    't_cdrg_r':'OBJECT1_CDRG_R',
    't_cdrg_t':'OBJECT1_CDRG_T',
    't_cdrg_n':'OBJECT1_CDRG_N',
    't_cdrg_rdot':'OBJECT1_CDRG_RDOT',
    't_cdrg_tdot':'OBJECT1_CDRG_TDOT',
    't_cdrg_ndot':'OBJECT1_CDRG_NDOT',
    't_cdrg_drg':'OBJECT1_CDRG_DRG',
    't_csrp_r':'OBJECT1_CSRP_R',
    't_csrp_t':'OBJECT1_CSRP_T',
    't_csrp_n':'OBJECT1_CSRP_N',
    't_csrp_rdot':'OBJECT1_CSRP_RDOT',
    't_csrp_tdot':'OBJECT1_CSRP_TDOT',
    't_csrp_ndot':'OBJECT1_CSRP_NDOT',
    't_csrp_drg':'OBJECT1_CSRP_DRG',
    't_csrp_srp':'OBJECT1_CSRP_SRP',
    't_cthr_r':'OBJECT1_CTHR_R',
    't_cthr_t':'OBJECT1_CTHR_T',
    't_cthr_n':'OBJECT1_CTHR_N',
    't_cthr_rdot':'OBJECT1_CTHR_RDOT',
    't_cthr_tdot':'OBJECT1_CTHR_TDOT',
    't_cthr_ndot':'OBJECT1_CTHR_NDOT',
    't_cthr_drg':'OBJECT1_CTHR_DRG',
    't_cthr_srp':'OBJECT1_CTHR_SRP',
    't_cthr_thr':'OBJECT1_CTHR_THR',
    'c_object_designator':'OBJECT2_OBJECT_DESIGNATOR',
    'c_catalog_name':'OBJECT2_CATALOG_NAME',
    'c_object_name':'OBJECT2_OBJECT_NAME',
    'c_international_designator':'OBJECT2_INTERNATIONAL_DESIGNATOR',
    'c_object_type':'OBJECT2_OBJECT_TYPE',
    'c_ephemeris_name':'OBJECT2_EPHEMERIS_NAME',
    'c_covariance_method':'OBJECT2_COVARIANCE_METHOD',
    'c_maneuverable':'OBJECT2_MANEUVERABLE',
    'c_orbit_center':'OBJECT2_ORBIT_CENTER',
    'c_ref_frame':'OBJECT2_REF_FRAME',
    'c_gravity_model':'OBJECT2_GRAVITY_MODEL',
    'c_atmospheric_model':'OBJECT2_ATMOSPHERIC_MODEL',
    'c_n_body_perturbations':'OBJECT2_N_BODY_PERTURBATIONS',
    'c_solar_rad_pressure':'OBJECT2_SOLAR_RAD_PRESSURE',
    'c_earth_tides':'OBJECT2_EARTH_TIDES',
    'c_intrack_thrust':'OBJECT2_INTRACK_THRUST',
    'c_time_lastob_start':'OBJECT2_TIME_LASTOB_START',
    'c_time_lastob_end':'OBJECT2_TIME_LASTOB_END',
    'c_recommended_od_span':'OBJECT2_RECOMMENDED_OD_SPAN',
    'c_actual_od_span':'OBJECT2_ACTUAL_OD_SPAN',
    'c_obs_available':'OBJECT2_OBS_AVAILABLE',
    'c_obs_used':'OBJECT2_OBS_USED',
    'c_tracks_available':'OBJECT2_TRACKS_AVAILABLE',
    'c_tracks_used':'OBJECT2_TRACKS_USED',
    'c_residuals_accepted':'OBJECT2_RESIDUALS_ACCEPTED',
    'c_weighted_rms':'OBJECT2_WEIGHTED_RMS',
    'c_area_pc':'OBJECT2_AREA_PC',
    'c_area_drg':'OBJECT2_AREA_DRG',
    'c_area_srg':'OBJECT2_AREA_SRP',
    'c_mass':'OBJECT2_MASS',
    'c_cd_area_over_mass':'OBJECT2_CD_AREA_OVER_MASS',
    'c_cr_area_over_mass':'OBJECT2_CR_AREA_OVER_MASS',
    'c_thrust_acceleration':'OBJECT2_THRUST_ACCELERATION',
    'c_sedr':'OBJECT2_SEDR',
    'c_x':'OBJECT2_X',
    'c_y':'OBJECT2_Y',
    'c_z':'OBJECT2_Z',
    'c_x_dot':'OBJECT2_X_DOT',
    'c_y_dot':'OBJECT2_Y_DOT',
    'c_z_dot':'OBJECT2_Z_DOT',
    'c_cr_r':'OBJECT2_CR_R',
    'c_ct_r':'OBJECT2_CT_R',
    'c_ct_t':'OBJECT2_CT_T',
    'c_cn_r':'OBJECT2_CN_R',
    'c_cn_t':'OBJECT2_CN_T',
    'c_cn_n':'OBJECT2_CN_N',
    'c_crdot_r':'OBJECT2_CRDOT_R',
    'c_crdot_t':'OBJECT2_CRDOT_T',
    'c_crdot_n':'OBJECT2_CRDOT_N',
    'c_crdot_rdot':'OBJECT2_CRDOT_RDOT',
    'c_ctdot_r':'OBJECT2_CTDOT_R',
    'c_ctdot_t':'OBJECT2_CTDOT_T',
    'c_ctdot_n':'OBJECT2_CTDOT_N',
    'c_ctdot_rdot':'OBJECT2_CTDOT_RDOT',
    'c_ctdot_tdot':'OBJECT2_CTDOT_TDOT',
    'c_cndot_r':'OBJECT2_CNDOT_R',
    'c_cndot_t':'OBJECT2_CNDOT_T',
    'c_cndot_n':'OBJECT2_CNDOT_N',
    'c_cndot_rdot':'OBJECT2_CNDOT_RDOT',
    'c_cndot_tdot':'OBJECT2_CNDOT_TDOT',
    'c_cndot_ndot':'OBJECT2_CNDOT_NDOT',
    'c_cdrg_r':'OBJECT2_CDRG_R',
    'c_cdrg_t':'OBJECT2_CDRG_T',
    'c_cdrg_n':'OBJECT2_CDRG_N',
    'c_cdrg_rdot':'OBJECT2_CDRG_RDOT',
    'c_cdrg_tdot':'OBJECT2_CDRG_TDOT',
    'c_cdrg_ndot':'OBJECT2_CDRG_NDOT',
    'c_cdrg_drg':'OBJECT2_CDRG_DRG',
    'c_csrp_r':'OBJECT2_CSRP_R',
    'c_csrp_t':'OBJECT2_CSRP_T',
    'c_csrp_n':'OBJECT2_CSRP_N',
    'c_csrp_rdot':'OBJECT2_CSRP_RDOT',
    'c_csrp_tdot':'OBJECT2_CSRP_TDOT',
    'c_csrp_ndot':'OBJECT2_CSRP_NDOT',
    'c_csrp_drg':'OBJECT2_CSRP_DRG',
    'c_csrp_srp':'OBJECT2_CSRP_SRP',
    'c_cthr_r':'OBJECT2_CTHR_R',
    'c_cthr_t':'OBJECT2_CTHR_T',
    'c_cthr_n':'OBJECT2_CTHR_N',
    'c_cthr_rdot':'OBJECT2_CTHR_RDOT',
    'c_cthr_tdot':'OBJECT2_CTHR_TDOT',
    'c_cthr_ndot':'OBJECT2_CTHR_NDOT',
    'c_cthr_drg':'OBJECT2_CTHR_DRG',
    'c_cthr_srp':'OBJECT2_CTHR_SRP',
    'c_cthr_thr':'OBJECT2_CTHR_THR'}, group_events_by='event_id', date_format='%Y-%m-%d %H:%M:%S.%f'):
    """
    Load EventDataset from a pandas DataFrame.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing CDM data
    cdm_compatible_fields : dict
        Mapping from DataFrame column names to CDM field names
    group_events_by : str
        Column name to group events by
    date_format : str
        Format string for parsing dates
        
    Returns
    -------
    EventDataset
        EventDataset constructed from the DataFrame
    """

    print('Dataframe with {} rows and {} columns'.format(len(df), len(df.columns)))
    print('Dropping columns with NaNs')
    df = df.dropna(axis=1)
    print('Dataframe with {} rows and {} columns'.format(len(df), len(df.columns)))
    pandas_column_names_after_dropping = list(df.columns)

    print('Grouping by {}'.format(group_events_by))
    df_events = df.groupby(group_events_by).groups
    print('Grouped into {} event(s)'.format(len(df_events)))
    events = []
    util.progress_bar_init('Converting DataFrame to EventDataset', len(df_events), 'Events')
    i = 0
    for k, v in df_events.items():
        util.progress_bar_update(i)
        i += 1
        df_event = df.iloc[v]
        cdms = []
        for _, df_cdm in df_event.iterrows():
            cdm = CDM()
            for pandas_name, cdm_name in cdm_compatible_fields.items():
                if pandas_name in pandas_column_names_after_dropping:
                    value = df_cdm[pandas_name]
                    # Check if the field is a date, if so transform to the correct date string format expected in the CCSDS 508.0-B-1 standard
                    if util.is_date(value, date_format):
                        value = util.transform_date_str(value, date_format, '%Y-%m-%dT%H:%M:%S.%f')
                    cdm[cdm_name] = value
            cdms.append(cdm)
        events.append(Event(cdms))
    util.progress_bar_end()
    event_dataset = EventDataset(events=events)
    print('\n{}'.format(event_dataset))
    return event_dataset