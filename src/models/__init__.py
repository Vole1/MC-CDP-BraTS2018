from enum import Enum


class NetType(Enum):
    cdo = 'Concrete dropout',
    cdp = 'Concrete droppath',
    mc_do = 'Monte-Carlo dropout',
    mc_df = 'Monte-Carlo dropfilter',
    sdo = 'Scheduled dropout'
    vanilla = 'vanilla',
