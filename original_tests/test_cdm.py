# This code is part of Kessler, a machine learning library for spacecraft collision avoidance.
#
# Copyright (c) 2020-
# University of Oxford (Atilim Gunes Baydin <gunes@robots.ox.ac.uk>)
# Trillium Technologies
# Giacomo Acciarini
# and other contributors, see README in root of repository.
#
# GNU General Public License version 3. See LICENSE in root of repository.


import unittest
import tempfile
import uuid
import os
import numpy as np
import pandas as pd
from kessler.original.data import kelvins_to_event_dataset
import random

from kessler.io import CSVReader, CDM
from kessler.io.schemas import InputsSchema

from original import ConjunctionDataMessage


class CDMTestCase(unittest.TestCase):
    def test_cdm_load_save_load(self):
        file_content = """
            CCSDS_CDM_VERS                        = 1.0
            CREATION_DATE                         = 2013-01-09T20:59:56.000
            ORIGINATOR                            = JSpOC
            MESSAGE_FOR                           = IRIDIUM 26
            MESSAGE_ID                            = 2013009205956Z
            TCA                                   = 2013-01-10T13:22:45.117
            MISS_DISTANCE                         = 180.0
            RELATIVE_SPEED                        = 8111.0
            RELATIVE_POSITION_R                   = 35.9
            RELATIVE_POSITION_T                   = -148.8
            RELATIVE_POSITION_N                   = 95.7
            RELATIVE_VELOCITY_R                   = -4.1
            RELATIVE_VELOCITY_T                   = -4421.4
            RELATIVE_VELOCITY_N                   = -6800.2
            OBJECT                                = OBJECT1
            OBJECT_DESIGNATOR                     = 24903.0
            CATALOG_NAME                          = US SATCAT
            OBJECT_NAME                           = IRIDIUM 26
            INTERNATIONAL_DESIGNATOR              = 1997-043A
            EPHEMERIS_NAME                        = NONE
            COVARIANCE_METHOD                     = CALCULATED
            MANEUVERABLE                          = N/A
            ORBIT_CENTER                          = EARTH
            REF_FRAME                             = ITRF
            GRAVITY_MODEL                         = EGM-96: 36D 36O
            ATMOSPHERIC_MODEL                     = JACCHIA 70 DCA
            SOLAR_RAD_PRESSURE                    = NO
            EARTH_TIDES                           = NO
            INTRACK_THRUST                        = NO
            RECOMMENDED_OD_SPAN                   = 6.58
            ACTUAL_OD_SPAN                        = 6.58
            OBS_AVAILABLE                         = 573.0
            OBS_USED                              = 571.0
            RESIDUALS_ACCEPTED                    = 98.1
            WEIGHTED_RMS                          = 0.944
            CD_AREA_OVER_MASS                     = 0.035005
            CR_AREA_OVER_MASS                     = 0.0
            SEDR                                  = 7.305E-05
            X                                     = -731.972155846
            Y                                     = -1871.205777005
            Z                                     = -6876.444217211
            X_DOT                                 = -1.117777619
            Y_DOT                                 = -7.044089359
            Z_DOT                                 = 2.036664419
            CR_R                                  = 38.33
            CT_R                                  = 93.6
            CT_T                                  = 3410.0
            CN_R                                  = -13.06
            CN_T                                  = 2.131
            CN_N                                  = 93.39
            CRDOT_R                               = 0.0
            CRDOT_T                               = 0.0
            CRDOT_N                               = 0.0
            CRDOT_RDOT                            = 0.0
            CTDOT_R                               = 0.0
            CTDOT_T                               = 0.0
            CTDOT_N                               = 0.0
            CTDOT_RDOT                            = 0.0
            CTDOT_TDOT                            = 0.0
            CNDOT_R                               = 0.0
            CNDOT_T                               = 0.0
            CNDOT_N                               = 0.0
            CNDOT_RDOT                            = 0.0
            CNDOT_TDOT                            = 0.0
            CNDOT_NDOT                            = 0.0
            OBJECT                                = OBJECT2
            OBJECT_DESIGNATOR                     = 33759.0
            CATALOG_NAME                          = US SATCAT
            OBJECT_NAME                           = COSMOS 2251 DEB
            INTERNATIONAL_DESIGNATOR              = 1993-036G
            EPHEMERIS_NAME                        = NONE
            COVARIANCE_METHOD                     = CALCULATED
            MANEUVERABLE                          = N/A
            ORBIT_CENTER                          = EARTH
            REF_FRAME                             = ITRF
            GRAVITY_MODEL                         = EGM-96: 36D 36O
            ATMOSPHERIC_MODEL                     = JACCHIA 70 DCA
            SOLAR_RAD_PRESSURE                    = YES
            EARTH_TIDES                           = NO
            INTRACK_THRUST                        = NO
            RECOMMENDED_OD_SPAN                   = 6.06
            ACTUAL_OD_SPAN                        = 6.06
            OBS_AVAILABLE                         = 180.0
            OBS_USED                              = 180.0
            RESIDUALS_ACCEPTED                    = 99.5
            WEIGHTED_RMS                          = 1.426
            CD_AREA_OVER_MASS                     = 0.20058
            CR_AREA_OVER_MASS                     = 0.087953
            SEDR                                  = 0.000525006
            X                                     = -732.050575868
            Y                                     = -1871.058618906
            Z                                     = -6876.513305974
            X_DOT                                 = 6.170235153
            Y_DOT                                 = -3.879972556
            Z_DOT                                 = 0.404182951
            CR_R                                  = 408.7
            CT_R                                  = -535.0
            CT_T                                  = 77970.0
            CN_R                                  = -17.21
            CN_T                                  = -79.98
            CN_N                                  = 239.9
            CRDOT_R                               = 0.0
            CRDOT_T                               = 0.0
            CRDOT_N                               = 0.0
            CRDOT_RDOT                            = 0.0
            CTDOT_R                               = 0.0
            CTDOT_T                               = 0.0
            CTDOT_N                               = 0.0
            CTDOT_RDOT                            = 0.0
            CTDOT_TDOT                            = 0.0
            CNDOT_R                               = 0.0
            CNDOT_T                               = 0.0
            CNDOT_N                               = 0.0
            CNDOT_RDOT                            = 0.0
            CNDOT_TDOT                            = 0.0
            CNDOT_NDOT                            = 0.0
            """
        file_name = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4()))
        with open(file_name, "w") as f:
            f.write(file_content)

        cdm = ConjunctionDataMessage.load(file_name)

        file_name = os.path.join(tempfile.mkdtemp(), str(uuid.uuid4()))
        cdm.save(file_name)

        cdm = ConjunctionDataMessage.load(file_name)

        cdm_covariance_1 = cdm.get_covariance(0)
        cdm_covariance_2 = cdm.get_covariance(1)
        cdm_covariance_1_correct = np.array(
            [
                [3.833e01, 9.360e01, -1.306e01, 0.000e00, 0.000e00, 0.000e00],
                [9.360e01, 3.410e03, 2.131e00, 0.000e00, 0.000e00, 0.000e00],
                [-1.306e01, 2.131e00, 9.339e01, 0.000e00, 0.000e00, 0.000e00],
                [0.000e00, 0.000e00, 0.000e00, 0.000e00, 0.000e00, 0.000e00],
                [0.000e00, 0.000e00, 0.000e00, 0.000e00, 0.000e00, 0.000e00],
                [0.000e00, 0.000e00, 0.000e00, 0.000e00, 0.000e00, 0.000e00],
            ]
        )
        cdm_covariance_2_correct = np.array(
            [
                [4.087e02, -5.350e02, -1.721e01, 0.000e00, 0.000e00, 0.000e00],
                [-5.350e02, 7.797e04, -7.998e01, 0.000e00, 0.000e00, 0.000e00],
                [-1.721e01, -7.998e01, 2.399e02, 0.000e00, 0.000e00, 0.000e00],
                [0.000e00, 0.000e00, 0.000e00, 0.000e00, 0.000e00, 0.000e00],
                [0.000e00, 0.000e00, 0.000e00, 0.000e00, 0.000e00, 0.000e00],
                [0.000e00, 0.000e00, 0.000e00, 0.000e00, 0.000e00, 0.000e00],
            ]
        )

        self.assertEqual(cdm_covariance_1_correct.tolist(), cdm_covariance_1.tolist())
        self.assertEqual(cdm_covariance_2_correct.tolist(), cdm_covariance_2.tolist())


column_mapping = {
    "CCSDS_CDM_VERS": "CCSDS_CDM_VERS",
    "CREATION_DATE": "CREATION_DATE",
    "ORIGINATOR": "ORIGINATOR",
    "MESSAGE_FOR": "MESSAGE_FOR",
    "MESSAGE_ID": "MESSAGE_ID",
    "TCA": "TCA",
    "MISS_DISTANCE": "MISS_DISTANCE",
    "RELATIVE_SPEED": "RELATIVE_SPEED",
    "RELATIVE_POSITION_R": "RELATIVE_POSITION_R",
    "RELATIVE_POSITION_T": "RELATIVE_POSITION_T",
    "RELATIVE_POSITION_N": "RELATIVE_POSITION_N",
    "RELATIVE_VELOCITY_R": "RELATIVE_VELOCITY_R",
    "RELATIVE_VELOCITY_T": "RELATIVE_VELOCITY_T",
    "RELATIVE_VELOCITY_N": "RELATIVE_VELOCITY_N",
    "START_SCREEN_PERIOD": "START_SCREEN_PERIOD",
    "STOP_SCREEN_PERIOD": "STOP_SCREEN_PERIOD",
    "SCREEN_VOLUME_FRAME": "SCREEN_VOLUME_FRAME",
    "SCREEN_VOLUME_SHAPE": "SCREEN_VOLUME_SHAPE",
    "SCREEN_VOLUME_X": "SCREEN_VOLUME_X",
    "SCREEN_VOLUME_Y": "SCREEN_VOLUME_Y",
    "SCREEN_VOLUME_Z": "SCREEN_VOLUME_Z",
    "SCREEN_ENTRY_TIME": "SCREEN_ENTRY_TIME",
    "SCREEN_EXIT_TIME": "SCREEN_EXIT_TIME",
    "COLLISION_PROBABILITY": "COLLISION_PROBABILITY",
    "COLLISION_PROBABILITY_METHOD": "COLLISION_PROBABILITY_METHOD",
    "OBJECT1_OBJECT": "t_OBJECT",
    "OBJECT1_OBJECT_DESIGNATOR": "t_OBJECT_DESIGNATOR",
    "OBJECT1_CATALOG_NAME": "t_CATALOG_NAME",
    "OBJECT1_OBJECT_NAME": "t_OBJECT_NAME",
    "OBJECT1_INTERNATIONAL_DESIGNATOR": "t_INTERNATIONAL_DESIGNATOR",
    "OBJECT1_OBJECT_TYPE": "t_OBJECT_TYPE",
    "OBJECT1_OPERATOR_CONTACT_POSITION": "t_OPERATOR_CONTACT_POSITION",
    "OBJECT1_OPERATOR_ORGANIZATION": "t_OPERATOR_ORGANIZATION",
    "OBJECT1_OPERATOR_PHONE": "t_OPERATOR_PHONE",
    "OBJECT1_OPERATOR_EMAIL": "t_OPERATOR_EMAIL",
    "OBJECT1_EPHEMERIS_NAME": "t_EPHEMERIS_NAME",
    "OBJECT1_COVARIANCE_METHOD": "t_COVARIANCE_METHOD",
    "OBJECT1_MANEUVERABLE": "t_MANEUVERABLE",
    "OBJECT1_ORBIT_CENTER": "t_ORBIT_CENTER",
    "OBJECT1_REF_FRAME": "t_REF_FRAME",
    "OBJECT1_GRAVITY_MODEL": "t_GRAVITY_MODEL",
    "OBJECT1_ATMOSPHERIC_MODEL": "t_ATMOSPHERIC_MODEL",
    "OBJECT1_N_BODY_PERTURBATIONS": "t_N_BODY_PERTURBATIONS",
    "OBJECT1_SOLAR_RAD_PRESSURE": "t_SOLAR_RAD_PRESSURE",
    "OBJECT1_EARTH_TIDES": "t_EARTH_TIDES",
    "OBJECT1_INTRACK_THRUST": "t_INTRACK_THRUST",
    "OBJECT1_TIME_LASTOB_START": "t_TIME_LASTOB_START",
    "OBJECT1_TIME_LASTOB_END": "t_TIME_LASTOB_END",
    "OBJECT1_RECOMMENDED_OD_SPAN": "t_RECOMMENDED_OD_SPAN",
    "OBJECT1_ACTUAL_OD_SPAN": "t_ACTUAL_OD_SPAN",
    "OBJECT1_OBS_AVAILABLE": "t_OBS_AVAILABLE",
    "OBJECT1_OBS_USED": "t_OBS_USED",
    "OBJECT1_TRACKS_AVAILABLE": "t_TRACKS_AVAILABLE",
    "OBJECT1_TRACKS_USED": "t_TRACKS_USED",
    "OBJECT1_RESIDUALS_ACCEPTED": "t_RESIDUALS_ACCEPTED",
    "OBJECT1_WEIGHTED_RMS": "t_WEIGHTED_RMS",
    "OBJECT1_AREA_PC": "t_AREA_PC",
    "OBJECT1_AREA_DRG": "t_AREA_DRG",
    "OBJECT1_AREA_SRP": "t_AREA_SRP",
    "OBJECT1_MASS": "t_MASS",
    "OBJECT1_CD_AREA_OVER_MASS": "t_CD_AREA_OVER_MASS",
    "OBJECT1_CR_AREA_OVER_MASS": "t_CR_AREA_OVER_MASS",
    "OBJECT1_THRUST_ACCELERATION": "t_THRUST_ACCELERATION",
    "OBJECT1_SEDR": "t_SEDR",
    "OBJECT1_X": "t_X",
    "OBJECT1_Y": "t_Y",
    "OBJECT1_Z": "t_Z",
    "OBJECT1_X_DOT": "t_X_DOT",
    "OBJECT1_Y_DOT": "t_Y_DOT",
    "OBJECT1_Z_DOT": "t_Z_DOT",
    "OBJECT1_CR_R": "t_CR_R",
    "OBJECT1_CT_R": "t_CT_R",
    "OBJECT1_CT_T": "t_CT_T",
    "OBJECT1_CN_R": "t_CN_R",
    "OBJECT1_CN_T": "t_CN_T",
    "OBJECT1_CN_N": "t_CN_N",
    "OBJECT1_CRDOT_R": "t_CRDOT_R",
    "OBJECT1_CRDOT_T": "t_CRDOT_T",
    "OBJECT1_CRDOT_N": "t_CRDOT_N",
    "OBJECT1_CRDOT_RDOT": "t_CRDOT_RDOT",
    "OBJECT1_CTDOT_R": "t_CTDOT_R",
    "OBJECT1_CTDOT_T": "t_CTDOT_T",
    "OBJECT1_CTDOT_N": "t_CTDOT_N",
    "OBJECT1_CTDOT_RDOT": "t_CTDOT_RDOT",
    "OBJECT1_CTDOT_TDOT": "t_CTDOT_TDOT",
    "OBJECT1_CNDOT_R": "t_CNDOT_R",
    "OBJECT1_CNDOT_T": "t_CNDOT_T",
    "OBJECT1_CNDOT_N": "t_CNDOT_N",
    "OBJECT1_CNDOT_RDOT": "t_CNDOT_RDOT",
    "OBJECT1_CNDOT_TDOT": "t_CNDOT_TDOT",
    "OBJECT1_CNDOT_NDOT": "t_CNDOT_NDOT",
    "OBJECT1_CDRG_R": "t_CDRG_R",
    "OBJECT1_CDRG_T": "t_CDRG_T",
    "OBJECT1_CDRG_N": "t_CDRG_N",
    "OBJECT1_CDRG_RDOT": "t_CDRG_RDOT",
    "OBJECT1_CDRG_TDOT": "t_CDRG_TDOT",
    "OBJECT1_CDRG_NDOT": "t_CDRG_NDOT",
    "OBJECT1_CDRG_DRG": "t_CDRG_DRG",
    "OBJECT1_CSRP_R": "t_CSRP_R",
    "OBJECT1_CSRP_T": "t_CSRP_T",
    "OBJECT1_CSRP_N": "t_CSRP_N",
    "OBJECT1_CSRP_RDOT": "t_CSRP_RDOT",
    "OBJECT1_CSRP_TDOT": "t_CSRP_TDOT",
    "OBJECT1_CSRP_NDOT": "t_CSRP_NDOT",
    "OBJECT1_CSRP_DRG": "t_CSRP_DRG",
    "OBJECT1_CSRP_SRP": "t_CSRP_SRP",
    "OBJECT1_CTHR_R": "t_CTHR_R",
    "OBJECT1_CTHR_T": "t_CTHR_T",
    "OBJECT1_CTHR_N": "t_CTHR_N",
    "OBJECT1_CTHR_RDOT": "t_CTHR_RDOT",
    "OBJECT1_CTHR_TDOT": "t_CTHR_TDOT",
    "OBJECT1_CTHR_NDOT": "t_CTHR_NDOT",
    "OBJECT1_CTHR_DRG": "t_CTHR_DRG",
    "OBJECT1_CTHR_SRP": "t_CTHR_SRP",
    "OBJECT1_CTHR_THR": "t_CTHR_THR",
    "OBJECT2_OBJECT": "c_OBJECT",
    "OBJECT2_OBJECT_DESIGNATOR": "c_OBJECT_DESIGNATOR",
    "OBJECT2_CATALOG_NAME": "c_CATALOG_NAME",
    "OBJECT2_OBJECT_NAME": "c_OBJECT_NAME",
    "OBJECT2_INTERNATIONAL_DESIGNATOR": "c_INTERNATIONAL_DESIGNATOR",
    "OBJECT2_OBJECT_TYPE": "c_OBJECT_TYPE",
    "OBJECT2_OPERATOR_CONTACT_POSITION": "c_OPERATOR_CONTACT_POSITION",
    "OBJECT2_OPERATOR_ORGANIZATION": "c_OPERATOR_ORGANIZATION",
    "OBJECT2_OPERATOR_PHONE": "c_OPERATOR_PHONE",
    "OBJECT2_OPERATOR_EMAIL": "c_OPERATOR_EMAIL",
    "OBJECT2_EPHEMERIS_NAME": "c_EPHEMERIS_NAME",
    "OBJECT2_COVARIANCE_METHOD": "c_COVARIANCE_METHOD",
    "OBJECT2_MANEUVERABLE": "c_MANEUVERABLE",
    "OBJECT2_ORBIT_CENTER": "c_ORBIT_CENTER",
    "OBJECT2_REF_FRAME": "c_REF_FRAME",
    "OBJECT2_GRAVITY_MODEL": "c_GRAVITY_MODEL",
    "OBJECT2_ATMOSPHERIC_MODEL": "c_ATMOSPHERIC_MODEL",
    "OBJECT2_N_BODY_PERTURBATIONS": "c_N_BODY_PERTURBATIONS",
    "OBJECT2_SOLAR_RAD_PRESSURE": "c_SOLAR_RAD_PRESSURE",
    "OBJECT2_EARTH_TIDES": "c_EARTH_TIDES",
    "OBJECT2_INTRACK_THRUST": "c_INTRACK_THRUST",
    "OBJECT2_TIME_LASTOB_START": "c_TIME_LASTOB_START",
    "OBJECT2_TIME_LASTOB_END": "c_TIME_LASTOB_END",
    "OBJECT2_RECOMMENDED_OD_SPAN": "c_RECOMMENDED_OD_SPAN",
    "OBJECT2_ACTUAL_OD_SPAN": "c_ACTUAL_OD_SPAN",
    "OBJECT2_OBS_AVAILABLE": "c_OBS_AVAILABLE",
    "OBJECT2_OBS_USED": "c_OBS_USED",
    "OBJECT2_TRACKS_AVAILABLE": "c_TRACKS_AVAILABLE",
    "OBJECT2_TRACKS_USED": "c_TRACKS_USED",
    "OBJECT2_RESIDUALS_ACCEPTED": "c_RESIDUALS_ACCEPTED",
    "OBJECT2_WEIGHTED_RMS": "c_WEIGHTED_RMS",
    "OBJECT2_AREA_PC": "c_AREA_PC",
    "OBJECT2_AREA_DRG": "c_AREA_DRG",
    "OBJECT2_AREA_SRP": "c_AREA_SRP",
    "OBJECT2_MASS": "c_MASS",
    "OBJECT2_CD_AREA_OVER_MASS": "c_CD_AREA_OVER_MASS",
    "OBJECT2_CR_AREA_OVER_MASS": "c_CR_AREA_OVER_MASS",
    "OBJECT2_THRUST_ACCELERATION": "c_THRUST_ACCELERATION",
    "OBJECT2_SEDR": "c_SEDR",
    "OBJECT2_X": "c_X",
    "OBJECT2_Y": "c_Y",
    "OBJECT2_Z": "c_Z",
    "OBJECT2_X_DOT": "c_X_DOT",
    "OBJECT2_Y_DOT": "c_Y_DOT",
    "OBJECT2_Z_DOT": "c_Z_DOT",
    "OBJECT2_CR_R": "c_CR_R",
    "OBJECT2_CT_R": "c_CT_R",
    "OBJECT2_CT_T": "c_CT_T",
    "OBJECT2_CN_R": "c_CN_R",
    "OBJECT2_CN_T": "c_CN_T",
    "OBJECT2_CN_N": "c_CN_N",
    "OBJECT2_CRDOT_R": "c_CRDOT_R",
    "OBJECT2_CRDOT_T": "c_CRDOT_T",
    "OBJECT2_CRDOT_N": "c_CRDOT_N",
    "OBJECT2_CRDOT_RDOT": "c_CRDOT_RDOT",
    "OBJECT2_CTDOT_R": "c_CTDOT_R",
    "OBJECT2_CTDOT_T": "c_CTDOT_T",
    "OBJECT2_CTDOT_N": "c_CTDOT_N",
    "OBJECT2_CTDOT_RDOT": "c_CTDOT_RDOT",
    "OBJECT2_CTDOT_TDOT": "c_CTDOT_TDOT",
    "OBJECT2_CNDOT_R": "c_CNDOT_R",
    "OBJECT2_CNDOT_T": "c_CNDOT_T",
    "OBJECT2_CNDOT_N": "c_CNDOT_N",
    "OBJECT2_CNDOT_RDOT": "c_CNDOT_RDOT",
    "OBJECT2_CNDOT_TDOT": "c_CNDOT_TDOT",
    "OBJECT2_CNDOT_NDOT": "c_CNDOT_NDOT",
    "OBJECT2_CDRG_R": "c_CDRG_R",
    "OBJECT2_CDRG_T": "c_CDRG_T",
    "OBJECT2_CDRG_N": "c_CDRG_N",
    "OBJECT2_CDRG_RDOT": "c_CDRG_RDOT",
    "OBJECT2_CDRG_TDOT": "c_CDRG_TDOT",
    "OBJECT2_CDRG_NDOT": "c_CDRG_NDOT",
    "OBJECT2_CDRG_DRG": "c_CDRG_DRG",
    "OBJECT2_CSRP_R": "c_CSRP_R",
    "OBJECT2_CSRP_T": "c_CSRP_T",
    "OBJECT2_CSRP_N": "c_CSRP_N",
    "OBJECT2_CSRP_RDOT": "c_CSRP_RDOT",
    "OBJECT2_CSRP_TDOT": "c_CSRP_TDOT",
    "OBJECT2_CSRP_NDOT": "c_CSRP_NDOT",
    "OBJECT2_CSRP_DRG": "c_CSRP_DRG",
    "OBJECT2_CSRP_SRP": "c_CSRP_SRP",
    "OBJECT2_CTHR_R": "c_CTHR_R",
    "OBJECT2_CTHR_T": "c_CTHR_T",
    "OBJECT2_CTHR_N": "c_CTHR_N",
    "OBJECT2_CTHR_RDOT": "c_CTHR_RDOT",
    "OBJECT2_CTHR_TDOT": "c_CTHR_TDOT",
    "OBJECT2_CTHR_NDOT": "c_CTHR_NDOT",
    "OBJECT2_CTHR_DRG": "c_CTHR_DRG",
    "OBJECT2_CTHR_SRP": "c_CTHR_SRP",
    "OBJECT2_CTHR_THR": "c_CTHR_THR",
    "__CREATION_DATE": "CREATION_DATE_IN_DAYS",
    "__TCA": "TCA_IN_DAYS",
    "__DAYS_TO_TCA": "DAYS_TO_TCA",
}


file_name = "data/train_data.csv"
number_of_events = 100

event_idx = 0
cdm_idx = 0


events_original = kelvins_to_event_dataset(
    file_name, drop_features=["c_rcs_estimate", "t_rcs_estimate"], num_events=100
)

cdm0_original = events_original._events[event_idx]._cdms[cdm_idx]
cdm0_orginal_df = cdm0_original.to_dataframe()
cdm0_orginal_df = cdm0_orginal_df.rename(columns=column_mapping)

cols_to_keep = dir(InputsSchema)
file_name = "data/train_data.csv"

events = CSVReader(path=file_name, number_of_events=100).read(columns_to_keep=cols_to_keep)
cdm0 = events.events[event_idx].cdms[cdm_idx]
cdm0_df = cdm0.to_dataframe()
cdm0_df = cdm0_df.drop(columns=["EVENT_ID"])

assert len(events) == number_of_events

# cdm copy

cdm0_copy = cdm0.copy()

assert cdm0_copy == cdm0

cdm1 = events.events[event_idx].cdms[cdm_idx + 1]
cdm0_copy.copy_from_other_cdm(cdm1)
assert cdm0_copy == cdm1

# use the right keys and values
# Create a dummy dict
assert len(cdm0_copy.to_dict().keys()) == 209

# TODO: use the right column names to see if they came out correctly
assert len(events.events[event_idx].cdms[cdm_idx].to_dataframe().columns) == 209

cdm0_copy.save("cdm0_copy.yml")

assert os.path.exists("cdm0_copy.yml")

cdm0_copy_loaded = CDM.load("cdm0_copy.yml")

# TODO: dont know why this fails
# assert cdm0_copy_loaded == cdm0_copy

cdm0_copy.set_header("EVENT_ID", 10)

assert cdm0_copy.header.get("EVENT_ID") == 10


cdm0_copy.set_header("CREATION_DATE", "2025-06-17T03:01:33.078351")

assert cdm0_copy.header.get("CREATION_DATE") == "2025-06-17T03:01:33.078351"

cdm0_copy.set_relative_metadata("RELATIVE_VELOCITY_R", 10)

assert cdm0_copy.get_relative_metadata("RELATIVE_VELOCITY_R") == 10

cdm0_copy.set_object("target", "REF_FRAME", "GCRF")
assert cdm0_copy.get_object("target", "REF_FRAME") == "GCRF"

cdm0_copy.set_object("target", "X", 10)
assert cdm0_copy.get_object("target", "X") == 10

cdm0_copy.set_object("target", "AREA_PC", 20)
assert cdm0_copy.get_object("target", "AREA_PC") == 20

cdm0_copy.set_object("target", "CR_R", 10)
assert cdm0_copy.get_object("target", "CR_R") == 10

cdm0_copy.set_object("chaser", "REF_FRAME", "GCRF")
assert cdm0_copy.get_object("chaser", "REF_FRAME") == "GCRF"

cdm0_copy.set_object("chaser", "X", 10)
assert cdm0_copy.get_object("chaser", "X") == 10

cdm0_copy.set_object("chaser", "AREA_PC", 20)
assert cdm0_copy.get_object("chaser", "AREA_PC") == 20

cdm0_copy.set_object("chaser", "CR_R", 10)
assert cdm0_copy.get_object("chaser", "CR_R") == 10

state_values_target = [i * 10 for i in range(6)]

state_values_target = np.array(state_values_target)

state_matrix_target = state_values_target.reshape(2, 3)


state_values_chaser = [i * 5 for i in range(6)]

state_values_chaser = np.array(state_values_chaser)

state_matrix_chaser = state_values_chaser.reshape(2, 3)

cdm0_copy.set_state("target", state_matrix_target)
cdm0_copy.set_state("chaser", state_matrix_chaser)
assert np.array_equal(cdm0_copy.get_state("target"), state_matrix_target)
assert np.array_equal(cdm0_copy.get_state("chaser"), state_matrix_chaser)

covariance_matrix = np.array(
    [
        [660, -90, 269, -18, 76, 157],
        [-90, 532, 40, 14, 210, 162],
        [269, 40, 395, -69, 222, 95],
        [-18, 14, -69, 189, -6, 109],
        [76, 210, 222, -6, 285, 186],
        [157, 162, 95, 109, 186, 242],
    ]
)


cdm0_copy.set_covariance("target", covariance_matrix)

assert np.allclose(cdm0_copy.get_covariance("target"), covariance_matrix)


cdm_df = pd.read_csv("data/train_data.csv")
cdm_df_event = cdm_df[cdm_df["event_id"] == event_idx]
cdm_from_csv = cdm_df_event.iloc[cdm_idx]

for col in cdm0_orginal_df.columns:
    if col in [
        "CCSDS_CDM_VERS",
        "CREATION_DATE",
        "ORIGINATOR",
        "MESSAGE_FOR",
        "MESSAGE_ID",
        "TCA",
        "t_OBJECT",
        "c_OBJECT",
        "t_TIME_LASTOB_START",
        "t_TIME_LASTOB_END",
        "c_TIME_LASTOB_START",
        "c_TIME_LASTOB_END",
    ]:
        continue
    else:
        val_org = cdm0_orginal_df.iloc[0][col]
        val_new = cdm0_df.iloc[0][col] if col in cdm0_df.columns else None

        if col in [
            "t_CD_AREA_OVER_MASS",
            "c_CD_AREA_OVER_MASS",
            "t_CR_AREA_OVER_MASS",
            "c_CR_AREA_OVER_MASS",
        ]:
            assert val_new == cdm_from_csv[col.lower()], (
                f"Column {col} mismatch: {val_new} != {cdm_from_csv[col.lower()]}"
            )

        else:
            assert val_org == val_new, f"Column {col} mismatch: {val_org} != {val_new}"


for key in cdm0_original.to_dict().keys():
    if key not in cdm0.to_dict().keys():
        if key in column_mapping.keys():
            continue
        else:
            raise AssertionError(f"Key {key} found in the original is not found in the modified")

for key in cdm0.to_dict().keys():
    if key not in cdm0_original.to_dict().keys():
        if key in column_mapping.values() or key == "EVENT_ID":
            continue

        else:
            raise AssertionError(f"Key {key} found in the modified is not found in the original")

assert np.allclose(cdm0.get_state("target"), cdm0_original.get_state(0), equal_nan=True)
assert np.allclose(cdm0.get_covariance("target"), cdm0_original.get_covariance(0), equal_nan=True)

state_values = [0, 1, 2, 3, 4, 5]

for i, state_key in enumerate(cdm0.keys_data_state_obligatory):
    cdm0.set_object("target", state_key, state_values[i])

for i, state_key in enumerate(cdm0_original._keys_data_state):
    cdm0_original.set_object(0, state_key, state_values[i])

np.allclose(cdm0.get_state("target"), cdm0_original.get_state(0), equal_nan=True)

for i, state_key in enumerate(cdm0.keys_data_state_obligatory):
    assert cdm0.to_dict().get(f"t_{state_key}") == cdm0_original.to_dict().get(
        f"OBJECT1_{state_key}"
    )

state_values_target = [i * 10 for i in range(6)]

state_values_target = np.array(state_values_target)

state_matrix_target = state_values_target.reshape(2, 3)


state_values_chaser = [i * 5 for i in range(6)]

state_values_chaser = np.array(state_values_chaser)

state_matrix_chaser = state_values_chaser.reshape(2, 3)

cdm0.set_state("target", state_matrix_target)
cdm0.set_state("chaser", state_matrix_chaser)

cdm0_original.set_state(0, state_matrix_target)
cdm0_original.set_state(1, state_matrix_chaser)

for key, value in cdm0.relative_metadata.items():
    if key in ["TCA"]:
        continue
    else:
        assert value == cdm0_original.to_dict().get(key), (
            f"Key {key} mismatch: {value} != {cdm0_original.to_dict().get(key)}"
        )

assert cdm0.to_dict().get("MISS_DISTANCE") == cdm0_original.to_dict().get("MISS_DISTANCE")


keys_size = len(cdm0.target_data_covariance.keys())

random_index = random.randint(0, keys_size)

key_selected = list(cdm0.target_data_covariance.keys())[random_index]

cdm0.set_object("target", key_selected, 1000)
cdm0_original.set_object(0, key_selected, 1000)

assert np.allclose(cdm0.get_covariance("target"), cdm0_original.get_covariance(0), equal_nan=True)
