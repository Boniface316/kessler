import datetime
from pydantic import BaseModel


class Header(BaseModel):
    CCSDS_CDM_VERS: str
    CREATION_DATE: datetime.datetime
    ORIGINATOR: str
    MESSAGE_FOR: str
    MESSAGE_ID: str


class RelativeMetadata(BaseModel):
    TCA: datetime.datetime
    MISS_DISTANCE: float
    RELATIVE_SPEED: float
    RELATIVE_POSITION_R: float
    RELATIVE_POSITION_T: float
    RELATIVE_POSITION_N: float
    RELATIVE_VELOCITY_R: float
    RELATIVE_VELOCITY_T: float
    RELATIVE_VELOCITY_N: float
    START_SCREEN_PERIOD: datetime.datetime
    STOP_SCREEN_PERIOD: datetime.datetime
    SCREEN_VOLUME_FRAME: str
    SCREEN_VOLUME_SHAPE: str
    SCREEN_VOLUME_X: float
    SCREEN_VOLUME_Y: float
    SCREEN_VOLUME_Z: float
    SCREEN_ENTRY_TIME: datetime.datetime
    SCREEN_EXIT_TIME: datetime.datetime
    COLLISION_PROBABILITY: float
    COLLISION_PROBABILITY_METHOD: str


class MetaData(BaseModel):
    OBJECT: str
    OBJECT_DESIGNATOR: str
    CATALOG_NAME: str
    OBJECT_NAME: str
    INTERNATIONAL_DESIGNATOR: str
    OBJECT_TYPE: str
    OPERATOR_CONTACT_POSITION: str
    OPERATOR_ORGANIZATION: str
    OPERATOR_PHONE: str
    OPERATOR_EMAIL: str
    EPHEMERIS_NAME: str
    COVARIANCE_METHOD: str
    MANEUVERABLE: str
    ORBIT_CENTER: str
    REF_FRAME: str
    GRAVITY_MODEL: str
    ATMOSPHERIC_MODEL: str
    N_BODY_PERTURBATIONS: str
    SOLAR_RAD_PRESSURE: str
    EARTH_TIDES: str
    INTRACK_THRUST: str


class OrbitDeterminationData(BaseModel):
    TIME_LASTOB_START: datetime.datetime
    TIME_LASTOB_END: datetime.datetime
    RECOMMENDED_OD_SPAN: float
    ACTUAL_OD_SPAN: float
    OBS_AVAILABLE: int
    OBS_USED: int
    TRACKS_AVAILABLE: int
    TRACKS_USED: int
    RESIDUALS_ACCEPTED: int
    WEIGHTED_RMS: float
    AREA_PC: float
    AREA_DRG: float
    AREA_SRP: float
    MASS: float
    CD_AREA_OVER_MASS: float
    CR_AREA_OVER_MASS: float
    THRUST_ACCELERATION: float
    SEDR: float


class StateVector(BaseModel):
    X: float
    Y: float
    Z: float
    X_DOT: float
    Y_DOT: float
    Z_DOT: float


class CovarianceMatrix(BaseModel):
    CR_R: float
    CT_R: float
    CT_T: float
    CN_R: float
    CN_T: float
    CN_N: float
    CRDOT_R: float
    CRDOT_T: float
    CRDOT_N: float
    CRDOT_RDOT: float
    CTDOT_R: float
    CTDOT_T: float
    CTDOT_N: float
    CTDOT_RDOT: float
    CTDOT_TDOT: float
    CNDOT_R: float
    CNDOT_T: float
    CNDOT_N: float
    CNDOT_RDOT: float
    CNDOT_TDOT: float
    CNDOT_NDOT: float
    CDRG_R: float
    CDRG_T: float
    CDRG_N: float
    CDRG_RDOT: float
    CDRG_TDOT: float
    CDRG_NDOT: float
    CDRG_DRG: float
    CSRP_R: float
    CSRP_T: float
    CSRP_N: float
    CSRP_RDOT: float
    CSRP_TDOT: float
    CSRP_NDOT: float
    CSRP_DRG: float
    CSRP_SRP: float
    CTHR_R: float
    CTHR_T: float
    CTHR_N: float
    CTHR_RDOT: float
    CTHR_TDOT: float
    CTHR_NDOT: float
    CTHR_DRG: float
    CTHR_SRP: float
    CTHR_THR: float


class ConjunctionDataMessage(BaseModel):
    HEADER: Header
    RELATIVE_METADATA: RelativeMetadata
    META_DATA: MetaData
    ORBIT_DETERMINATION_DATA: OrbitDeterminationData
    STATE_VECTOR: StateVector
    COVARIANCE_MATRIX: CovarianceMatrix

    # if file_name:
    #         self.copy_from(ConjunctionDataMessage.load(file_name))

    def copy(self):
        return ConjunctionDataMessage(
            HEADER=self.HEADER.copy(),
            RELATIVE_METADATA=self.RELATIVE_METADATA.copy(),
            META_DATA=self.META_DATA.copy(),
            ORBIT_DETERMINATION_DATA=self.ORBIT_DETERMINATION_DATA.copy(),
            STATE_VECTOR=self.STATE_VECTOR.copy(),
            COVARIANCE_MATRIX=self.COVARIANCE_MATRIX.copy(),
        )

    def copy_from(self, other_cdm):
        self.HEADER = other_cdm.HEADER.copy()
        self.RELATIVE_METADATA = other_cdm.RELATIVE_METADATA.copy()
        self.META_DATA = other_cdm.META_DATA.copy()
        self.ORBIT_DETERMINATION_DATA = other_cdm.ORBIT_DETERMINATION_DATA.copy()
        self.STATE_VECTOR = other_cdm.STATE_VECTOR.copy()
        self.COVARIANCE_MATRIX = other_cdm.COVARIANCE_MATRIX.copy()
