from models.unets_do import densenet_fpn, nasnet_cdo_fpn, nasnet_cdp_fpn, nasnet_df_fpn, nasnet_do_fpn,\
    nasnet_sdo_fpn, xception_fpn


def make_model(network, input_shape, pretrained_weights, do_rate=0.3, **kwargs):
    if network == 'densenet169':
        return densenet_fpn(input_shape, channels=2, weights=pretrained_weights, activation="sigmoid")
    elif network == 'nasnet_cdo':
        return nasnet_cdo_fpn(input_shape, channels=2, weights=pretrained_weights, activation="sigmoid")
    elif network == 'nasnet_cdp':
        return nasnet_cdp_fpn(input_shape, channels=2, weights=pretrained_weights, activation="sigmoid")
    elif network == 'nasnet_do':
        return nasnet_do_fpn(input_shape, channels=2, do_rate=do_rate, weights=pretrained_weights, activation="sigmoid")
    elif network == 'nasnet_df':
        return nasnet_df_fpn(input_shape, channels=2, do_rate=do_rate, weights=pretrained_weights, activation="sigmoid")
    elif network == 'nasnet_sch_dp':
        return nasnet_sdo_fpn(input_shape, channels=2, do_rate=do_rate, weights=pretrained_weights, activation="sigmoid",
                              **kwargs)
    elif network == 'xception':
        return xception_fpn(input_shape, channels=2, weights=pretrained_weights, activation="sigmoid")
    else:
        raise ValueError('unknown network ' + network)
