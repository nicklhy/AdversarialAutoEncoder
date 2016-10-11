import mxnet as mx


def make_aae_sym(data_dim=784, n_dim=2, n_encoder=1000, n_decoder=1000, n_discriminator=500, enc_mult=1, dec_mult=1, dis_mult=1, with_bn=True, supervised=False):
    data = mx.sym.Variable('data')
    data = mx.sym.Flatten(data=data)
    z = mx.sym.Variable('z')
    z = mx.sym.Flatten(data=z)

    #  Encoder
    enc = mx.sym.FullyConnected(data=data, num_hidden=n_encoder, attr={'lr_mult': str(enc_mult)}, name='enc_fc1')
    if with_bn:
        enc = mx.sym.BatchNorm(data=enc, name='enc_bn1')
    enc = mx.sym.Activation(data=enc, name='enc_relu1', act_type='relu')
    enc = mx.sym.FullyConnected(data=enc, num_hidden=n_encoder, attr={'lr_mult': str(enc_mult)}, name='enc_fc2')
    if with_bn:
        enc = mx.sym.BatchNorm(data=enc, name='enc_bn2')
    enc = mx.sym.Activation(data=enc, name='enc_relu2', act_type='relu')
    enc = mx.sym.FullyConnected(data=enc, num_hidden=n_dim, attr={'lr_mult': str(enc_mult)}, name='enc_fc3')
    if with_bn:
        enc = mx.sym.BatchNorm(data=enc, name='enc_bn3')

    #  Decoder
    dec = mx.sym.FullyConnected(data=z, num_hidden=n_decoder, attr={'lr_mult': str(dec_mult)}, name='dec_fc1')
    if with_bn:
        dec = mx.sym.BatchNorm(data=dec, name='enc_bn1')
    dec = mx.sym.Activation(data=dec, name='dec_relu1', act_type='relu')
    dec = mx.sym.FullyConnected(data=dec, num_hidden=n_decoder, attr={'lr_mult': str(dec_mult)}, name='dec_fc2')
    if with_bn:
        dec = mx.sym.BatchNorm(data=dec, name='enc_bn2')
    dec = mx.sym.Activation(data=dec, name='dec_relu2', act_type='relu')
    dec = mx.sym.FullyConnected(data=dec, num_hidden=data_dim, attr={'lr_mult': str(dec_mult)}, name='dec_fc3')
    dec = mx.sym.Activation(data=dec, name='dec_out', act_type='sigmoid')
    dec = mx.sym.LinearRegressionOutput(data=dec, label=data, name='dec_loss')

    #  Discriminator
    label_pq = mx.sym.Variable('label_pq')
    if supervised:
        label_n = mx.sym.Variable('label_n')
        z = mx.sym.Concat(label_n, z)

    dis = mx.sym.FullyConnected(data=z, num_hidden=n_discriminator, attr={'lr_mult': str(dis_mult)}, name='dis_fc1')
    if with_bn:
        dis = mx.sym.BatchNorm(data=dis, name='dis_bn1')
    dis = mx.sym.Activation(data=dis, act_type='relu', name='dis_relu1')
    dis = mx.sym.FullyConnected(data=dis, num_hidden=n_discriminator, attr={'lr_mult': str(dis_mult)}, name='dis_fc2')
    if with_bn:
        dis = mx.sym.BatchNorm(data=dis, name='dis_bn2')
    dis = mx.sym.Activation(data=dis, act_type='relu', name='dis_relu1')
    dis = mx.sym.FullyConnected(data=dis, num_hidden=1, attr={'lr_mult': str(dis_mult)}, name='dis_clf')
    dis = mx.sym.LogisticRegressionOutput(data=dis, name='dis_pred', label=label_pq)

    return enc, dec, dis

