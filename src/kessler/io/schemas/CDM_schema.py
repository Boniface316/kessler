import pandera.typing as papd
import pandera as pa
import pandera.typing.common as padt
from .._base import Schema


class InputsSchema(Schema):
    """Schema for the project inputs."""

    index: papd.Index[padt.UInt32] = pa.Field(ge=0)
    event_id: papd.Series[padt.Int64] = pa.Field(ge=0, le=13153)
    time_to_tca: papd.Series[padt.Float32] = pa.Field(ge=-0.0, le=7.0)
    mission_id: papd.Series[padt.Int64] = pa.Field(ge=1)
    risk: papd.Series[padt.Float16] = pa.Field(ge=-100.0, le=100.0)
    max_risk_estimate: papd.Series[padt.Float32] = pa.Field(ge=-100.0, le=100.00)
    max_risk_scaling: papd.Series[padt.Float64] = pa.Field(ge=0.0)
    miss_distance: papd.Series[padt.Int32] = pa.Field(ge=0)
    relative_speed: papd.Series[padt.Int32] = pa.Field(ge=0)
    relative_position_r: papd.Series[padt.Float32] = pa.Field()
    relative_position_t: papd.Series[padt.Float32] = pa.Field()
    relative_position_n: papd.Series[padt.Float32] = pa.Field()
    relative_velocity_r: papd.Series[padt.Float32] = pa.Field()
    relative_velocity_t: papd.Series[padt.Float32] = pa.Field()
    relative_velocity_n: papd.Series[padt.Float32] = pa.Field()
    t_time_lastob_start: papd.Series[padt.Float32] = pa.Field(ge=0.0)
    t_time_lastob_end: papd.Series[padt.Float32] = pa.Field(ge=0.0)
    t_recommended_od_span: papd.Series[padt.Float32] = pa.Field(ge=0.0)
    t_actual_od_span: papd.Series[padt.Float32] = pa.Field(ge=0.0)
    t_obs_available: papd.Series[padt.Int64] = pa.Field(ge=0.0)
    t_obs_used: papd.Series[padt.Int32] = pa.Field(ge=0)
    t_residuals_accepted: papd.Series[padt.Float32] = pa.Field(ge=0.0, le=100.0)
    t_weighted_rms: papd.Series[padt.Float16] = pa.Field(ge=0.0)
    # t_rcs_estimate: papd.Series[padt.Float32] = pa.Field(ge=0.0)
    t_cd_area_over_mass: papd.Series[padt.Float32] = pa.Field(ge=-10.0, le=10.0)
    t_cr_area_over_mass: papd.Series[padt.Float32] = pa.Field(ge=00.0, le=10.0)
    t_sedr: papd.Series[padt.Float32] = pa.Field(ge=-1.0, le=1.0)
    t_j2k_sma: papd.Series[padt.Float64] = pa.Field(ge=0.0)
    t_j2k_ecc: papd.Series[padt.Float32] = pa.Field(ge=0.0)
    t_j2k_inc: papd.Series[padt.Float32] = pa.Field(ge=0.0)
    t_ct_r: papd.Series[padt.Float32] = pa.Field(ge=-2.0, le=2.0)
    t_cn_r: papd.Series[padt.Float32] = pa.Field(ge=-2.0, le=2.0)
    t_cn_t: papd.Series[padt.Float32] = pa.Field(ge=-2.0, le=2.0)
    t_crdot_r: papd.Series[padt.Float32] = pa.Field(ge=-2.0, le=2.0)
    t_crdot_t: papd.Series[padt.Float32] = pa.Field(ge=-2.0, le=2.0)
    t_crdot_n: papd.Series[padt.Float32] = pa.Field(ge=-2.0, le=2.0)
    t_ctdot_r: papd.Series[padt.Float32] = pa.Field(ge=-2.0, le=2.0)
    t_ctdot_t: papd.Series[padt.Float32] = pa.Field(ge=-2.0, le=2.0)
    t_ctdot_n: papd.Series[padt.Float32] = pa.Field(ge=-2.0, le=2.0)
    t_ctdot_rdot: papd.Series[padt.Float32] = pa.Field(ge=-1.0, le=1.0)
    t_cndot_r: papd.Series[padt.Float32] = pa.Field(ge=-1.0, le=1.0)
    t_cndot_t: papd.Series[padt.Float32] = pa.Field(ge=-1.0, le=1.0)
    t_cndot_n: papd.Series[padt.Float32] = pa.Field(ge=-1.0, le=1.0)
    t_cndot_rdot: papd.Series[padt.Float32] = pa.Field(ge=-1.0, le=1.0)
    t_cndot_tdot: papd.Series[padt.Float32] = pa.Field(ge=-1.0, le=1.0)
    c_object_type: papd.Series[str] = pa.Field(
        isin=["UNKNOWN", "DEBRIS", "PAYLOAD", "ROCKET BODY", "TBA"]
    )
    c_time_lastob_start: papd.Series[padt.Float32] = pa.Field(ge=0.0)
    c_time_lastob_end: papd.Series[padt.Float32] = pa.Field(ge=0.0)
    c_recommended_od_span: papd.Series[padt.Float32] = pa.Field(ge=0.0)
    c_actual_od_span: papd.Series[padt.Float32] = pa.Field(ge=0.0)
    c_obs_available: papd.Series[padt.Int64] = pa.Field(ge=0)
    c_obs_used: papd.Series[padt.Int64] = pa.Field(ge=0)
    c_residuals_accepted: papd.Series[padt.Float32] = pa.Field(ge=0.0, le=100.0)
    c_weighted_rms: papd.Series[padt.Float32] = pa.Field(ge=0.0)
    # c_rcs_estimate: papd.Series[padt.Float32] = pa.Field(ge=0.0)
    c_cd_area_over_mass: papd.Series[padt.Float32] = pa.Field(ge=-200.0, le=200.0)
    c_cr_area_over_mass: papd.Series[padt.Float32] = pa.Field(ge=-100.0, le=100.0)
    c_sedr: papd.Series[padt.Float32] = pa.Field(ge=-1.0, le=1.0)
    c_j2k_sma: papd.Series[padt.Float64] = pa.Field(ge=0.0)
    c_j2k_ecc: papd.Series[padt.Float32] = pa.Field(ge=0.0)
    c_j2k_inc: papd.Series[padt.Float32] = pa.Field(ge=0.0)
    c_ct_r: papd.Series[padt.Float32] = pa.Field(ge=-2.0, le=2.0)
    c_cn_r: papd.Series[padt.Float32] = pa.Field(ge=-1.0, le=1.0)
    c_cn_t: papd.Series[padt.Float32] = pa.Field(ge=-1.0, le=1.0)
    c_crdot_r: papd.Series[padt.Float32] = pa.Field(ge=-2.0, le=2.0)
    c_crdot_t: papd.Series[padt.Float32] = pa.Field(ge=-2.0, le=2.0)
    c_crdot_n: papd.Series[padt.Float32] = pa.Field(ge=-2.0, le=2.0)
    c_ctdot_r: papd.Series[padt.Float32] = pa.Field(ge=-2.0, le=2.0)
    c_ctdot_t: papd.Series[padt.Float32] = pa.Field(ge=-2.0, le=2.0)
    c_ctdot_n: papd.Series[padt.Float32] = pa.Field(ge=-2.0, le=2.0)
    c_ctdot_rdot: papd.Series[padt.Float32] = pa.Field(ge=-1.0, le=1.0)
    c_cndot_r: papd.Series[padt.Float32] = pa.Field(ge=-1.0, le=1.0)
    c_cndot_t: papd.Series[padt.Float32] = pa.Field(ge=-1.0, le=1.0)
    c_cndot_n: papd.Series[padt.Float32] = pa.Field(ge=-1.0, le=1.0)
    c_cndot_rdot: papd.Series[padt.Float32] = pa.Field(ge=-1.0, le=1.0)
    c_cndot_tdot: papd.Series[padt.Float32] = pa.Field(ge=-1.0, le=1.0)
    t_span: papd.Series[padt.Float32] = pa.Field(ge=0.0)
    c_span: papd.Series[padt.Float32] = pa.Field(ge=0.0)
    t_h_apo: papd.Series[padt.Float64] = pa.Field(ge=0.0)
    t_h_per: papd.Series[padt.Float64] = pa.Field(ge=0.0)
    c_h_apo: papd.Series[padt.Float64] = pa.Field(ge=0.0)
    c_h_per: papd.Series[padt.Float64] = pa.Field(ge=0.0)
    geocentric_latitude: papd.Series[padt.Float32] = pa.Field(ge=-100.0, le=100.0)
    azimuth: papd.Series[padt.Float32] = pa.Field(ge=-360.0, le=360.0)
    elevation: papd.Series[padt.Float32] = pa.Field(ge=-90.0, le=90.0)
    mahalanobis_distance: papd.Series[padt.Float64] = pa.Field(ge=0.0)
    t_position_covariance_det: papd.Series[padt.Float64] = pa.Field()
    c_position_covariance_det: papd.Series[padt.Float64] = pa.Field()
    t_sigma_r: papd.Series[padt.Float64] = pa.Field(ge=0.0)
    c_sigma_r: papd.Series[padt.Float64] = pa.Field(ge=0.0)
    t_sigma_t: papd.Series[padt.Float64] = pa.Field(ge=0.0)
    c_sigma_t: papd.Series[padt.Float64] = pa.Field(ge=0.0)
    t_sigma_n: papd.Series[padt.Float64] = pa.Field(ge=0.0)
    c_sigma_n: papd.Series[padt.Float64] = pa.Field(ge=0.0)
    t_sigma_rdot: papd.Series[padt.Float64] = pa.Field(ge=0.0)
    c_sigma_rdot: papd.Series[padt.Float64] = pa.Field(ge=0.0)
    t_sigma_tdot: papd.Series[padt.Float64] = pa.Field(ge=0.0)
    c_sigma_tdot: papd.Series[padt.Float64] = pa.Field(ge=0.0)
    t_sigma_ndot: papd.Series[padt.Float64] = pa.Field(ge=0.0)
    c_sigma_ndot: papd.Series[padt.Float64] = pa.Field(ge=0.0)
    F10: papd.Series[padt.Int64] = pa.Field(ge=0)
    F3M: papd.Series[padt.Int64] = pa.Field(ge=0)
    SSN: papd.Series[padt.Int64] = pa.Field(ge=0)
    AP: papd.Series[padt.Int64] = pa.Field(ge=0)


class ExampleTargetsSchema(Schema):
    """Schema for the project target.

    Example:
        instant: papd.Index[padt.UInt32] = pa.Field(ge=0)
        cnt: papd.Series[padt.UInt32] = pa.Field(ge=0)
    """

    index: papd.Index[padt.UInt32] = pa.Field(ge=0)
    target: papd.Series[padt.UInt32] = pa.Field(ge=0)


class ExampleOutputsSchema(Schema):
    """Schema for the project output

    Example:
        instant: papd.Index[padt.UInt32] = pa.Field(ge=0)
        prediction: papd.Series[padt.UInt32] = pa.Field(ge=0)

    """

    index: papd.Index[padt.UInt32] = pa.Field(ge=0)
    prediction: papd.Series[padt.UInt32] = pa.Field(ge=0)


class ExampleSHAPValuesSchema(Schema):
    """Schema for the project shap values."""

    class Config:
        """Default configurations this schema.

        Parameters:
            dtype (str): dataframe default data type.
            strict (bool): ensure the data type is correct.
        """

        dtype: str = "float32"
        strict: bool = False


class ExampleFeatureImportancesSchema(Schema):
    """Schema for the project feature importances."""

    feature: papd.Series[str] = pa.Field()
    importance: papd.Series[padt.Float32] = pa.Field()
