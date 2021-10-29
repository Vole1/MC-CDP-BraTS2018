from enum import Enum


class NetType(Enum):
    cdo = 'Monte-Carlo Concrete Dropout',
    cdp = 'Monte-Carlo Concrete Droppath',
    mc_do = 'Monte-Carlo Dropout',
    mc_df = 'Monte-Carlo Dropfilter',
    sdo = 'Monte-Carlo Scheduled Dropout'
    sdp = 'Monte-Carlo Scheduled Droppath'
    vanilla = 'vanilla',
