from pydantic import BaseModel
from copy import deepcopy


class ConjuctionDataMessage(BaseModel):
    header: dict
    relative_metadata: dict

    target_metadata: dict
    target_data_od: dict
    target_data_state: dict
    target_data_covariance: dict

    chaser_metadata: dict
    chaser_data_od: dict
    chaser_data_state: dict
    chaser_data_covariance: dict

    def copy(self):
        return ConjuctionDataMessage(
            header=deepcopy(self.header),
            relative_metadata=deepcopy(self.relative_metadata),
            target_metadata=deepcopy(self.target_metadata),
            target_data_od=deepcopy(self.target_data_od),
            target_data_state=deepcopy(self.target_data_state),
            target_data_covariance=deepcopy(self.target_data_covariance),
            chaser_metadata=deepcopy(self.chaser_metadata),
            chaser_data_od=deepcopy(self.chaser_data_od),
            chaser_data_state=deepcopy(self.chaser_data_state),
            chaser_data_covariance=deepcopy(self.chaser_data_covariance),
        )

    def copy_from_other_cdm(sefl, other_cdm):
        pass

    def to_dict(self):
        pass

    def to_dataframe(self):
        pass

    def save(self):
        pass

    def __hash__(self):
        pass

    def __eq__(self, value):
        pass

    def set_header(self, key, value):
        pass

    def set_relative_metadata(self, key, value):
        pass

    def set_object(self, key, value):
        pass

    def get_object(self, object_id, key):
        pass

    def get_relative_metadata(self, key):
        pass

    def set_state(self, object_id, state):
        pass

    def _update_miss_distance(self):
        pass

    def _update_state_relative(self):
        pass

    def get_state_relative(self):
        pass

    def get_state(self, object_id):
        pass

    def get_covariance(self, object_id):
        pass

    def set_covariance(self, object_id, covariance):
        pass

    def validate(self):
        pass

    def key_value_notation(self):
        ret = "\n\n"
        ret += "\n".join([f"{k}: {v}" for k, v in self.header.items()])
        ret += "\n\n"
        ret += "\n".join([f"{k}: {v}" for k, v in self.relative_metadata.items()])
        ret += "\n\n"
        ret += "\n".join([f"{k}: {v}" for k, v in self.target_metadata.items()])
        ret += "\n\n Target Data: \n"
        ret += "\n".join([f"{k}: {v}" for k, v in self.target_data_od.items()])
        ret += "\n".join([f"{k}: {v}" for k, v in self.target_data_state.items()])
        ret += "\n".join([f"{k}: {v}" for k, v in self.target_data_covariance.items()])
        ret += "\n\n Chaser Data: \n"
        ret += "\n".join([f"{k}: {v}" for k, v in self.chaser_metadata.items()])
        ret += "\n".join([f"{k}: {v}" for k, v in self.chaser_data_od.items()])
        ret += "\n".join([f"{k}: {v}" for k, v in self.chaser_data_state.items()])
        ret += "\n".join([f"{k}: {v}" for k, v in self.chaser_data_covariance.items()])
        return ret

    def __repr__(self):
        return self.key_value_notation()

    def __getitem__(self, key):
        return self.to_dict()[key]

    def __setitem__(self, key, value):
        pass


CDM = ConjuctionDataMessage
