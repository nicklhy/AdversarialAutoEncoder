import os
import sys
import mxnet as mx
import numpy as np
from data_factory import RandIter, get_mnist
from model import make_aae_sym


if __name__ == '__main__':
    # =============setting============
    ctx = mx.gpu(3)
    display_step = 20
    epoch_num = 40
    check_point = 20
    early_stop = True
    dataset = 'mnist'
    n_labels = 10
    z_prior = 'swiss_roll'
    if z_prior == 'uniform':
        z_args = {
            'minv': -2.0,
            'maxv': 2.0
        }
    elif z_prior == 'gaussian_mixture':
        z_args = {
            'x_var': 0.5,
            'y_var': 0.1
        }
    elif z_prior == 'swiss_roll':
        z_args = {}
    elif z_prior == 'gaussian':
        raise ValueError, 'please use gaussian mixture for supervised training'
    else:
        raise ValueError, 'unknown z_prior'
    batch_size=100
    n_dim = 2
    n_encoder = 1000
    n_decoder = 1000
    n_discriminator = 500
    enc_mult = 1
    dec_mult = 1
    dis_mult = 1
    std = 0.01
    lr = 0.002
    lr_factor = 1
    lr_factor_step = 2000
    wd = 0.0
    beta1 = 0.1
    optimizer_args = {
        'optimizer': 'adam',
        'optimizer_params': {
            'clip_gradient': 5.0,
            'learning_rate': lr,
            'lr_scheduler': mx.lr_scheduler.FactorScheduler(lr_factor_step, lr_factor),
            'wd': wd,
            'beta1': beta1,
        }
    }

    if not os.path.exists('cache'):
        os.mkdir('cache')

    if dataset == 'mnist':
        X_train, X_test, Y_train, Y_test = get_mnist(root_dir='./cache', train_ratio=0.9)
        train_iter = mx.io.NDArrayIter(X_train, label=Y_train, batch_size=batch_size)
        #  change the label name from softmax_label to n
        train_iter.label = [('label_n', train_iter.label[0][1])]
    else:
        raise NotImplementedError
    rand_iter = RandIter(batch_size, n_dim, z_prior=z_prior, n_labels=n_labels, with_label=True, **z_args)
    label_pq = mx.nd.zeros((batch_size,), ctx=ctx)

    sym_enc, sym_dec, sym_dis = make_aae_sym(data_dim=784,
                                             n_dim=n_dim,
                                             n_encoder=n_encoder,
                                             n_decoder=n_decoder,
                                             n_discriminator=n_discriminator,
                                             enc_mult=enc_mult,
                                             dec_mult=dec_mult,
                                             dis_mult=dis_mult,
                                             with_bn=False,
                                             supervised=True)

    mod_enc = mx.mod.Module(symbol=sym_enc, data_names=('data',), label_names=(), context=ctx)
    mod_enc.bind(data_shapes=train_iter.provide_data,
                label_shapes=[],
                inputs_need_grad=False)
    mod_enc.init_params(initializer=mx.init.Normal(std))
    mod_enc.init_optimizer(**optimizer_args)

    mod_dec = mx.mod.Module(symbol=sym_dec, data_names=('z',), label_names=('data',), context=ctx)
    mod_dec.bind(data_shapes=rand_iter.provide_data,
                 label_shapes=train_iter.provide_data,
                 inputs_need_grad=True)
    mod_dec.init_params(initializer=mx.init.Normal(std))
    mod_dec.init_optimizer(**optimizer_args)

    mod_dis = mx.mod.Module(symbol=sym_dis, data_names=('z', 'label_n'), label_names=('label_pq',), context=ctx)
    mod_dis.bind(data_shapes=rand_iter.provide_data+rand_iter.provide_label,
                 label_shapes=[('label_pq', (batch_size,))],
                 inputs_need_grad=True)
    mod_dis.init_params(initializer=mx.init.Normal(std))
    mod_dis.init_optimizer(**optimizer_args)

    def facc(label, pred):
        pred = pred.ravel()
        label = label.ravel()
        return ((pred > 0.5) == label).mean()

    def fentropy(label, pred):
        pred = pred.ravel()
        label = label.ravel()
        return -(label*np.log(pred+1e-12) + (1.-label)*np.log(1.-pred+1e-12)).mean()

    def frmse(label, pred):
        dim = label.size/label.shape[0]
        label = label.reshape((-1, dim))
        pred = pred.reshape((-1, dim))
        return np.linalg.norm(label-pred, axis=1).mean()

    metric_dec_rmse = mx.metric.CustomMetric(frmse)
    metric_dis_accuracy = mx.metric.CustomMetric(facc)
    metric_dis_entropy = mx.metric.CustomMetric(fentropy)
    metric_fool_dis_accuracy = mx.metric.CustomMetric(facc)
    metric_fool_dis_entropy = mx.metric.CustomMetric(fentropy)
    #  metric_dis_accuracy = mx.metric.create('accuracy')
    #  metric_dis_entropy = mx.metric.create('ce')
    #  metric_fool_dis_accuracy = mx.metric.create('accuracy')
    #  metric_fool_dis_entropy = mx.metric.create('ce')

    print 'Training ...'
    for epoch in xrange(epoch_num):
        train_iter.reset()
        for t, batch in enumerate(train_iter):
            batch_label_one_hot = np.zeros((batch_size, n_labels), dtype=np.float32)
            batch_label_np = batch.label[0].asnumpy()
            for i in xrange(batch_size):
                batch_label_one_hot[i, int(batch_label_np[i])] = 1
            batch_label_one_hot = mx.nd.array(batch_label_one_hot)

            rbatch = rand_iter.next()

            #  reconstruction phase: update encoder and decoder
            mod_enc.forward(batch, is_train=True)
            qz = mod_enc.get_outputs()
            mod_dec.forward(mx.io.DataBatch(qz, batch.data), is_train=True)
            mod_dec.backward()
            diff_dec = mod_dec.get_input_grads()
            mod_enc.backward(diff_dec)
            mod_enc.update()
            mod_dec.update()

            metric_dec_rmse.update(batch.data, mod_dec.get_outputs())

            #  regularization phase
            #  step 1: update discriminator
            label_pq[:] = 0
            mod_dis.forward(mx.io.DataBatch(qz+[batch_label_one_hot], [label_pq]), is_train=True)
            mod_dis.backward()
            #  mod_dis.update()
            gradD = [[grad.copyto(grad.context) for grad in grads] for grads in mod_dis._exec_group.grad_arrays]
            metric_dis_accuracy.update([label_pq], mod_dis.get_outputs())
            metric_dis_entropy.update([label_pq], mod_dis.get_outputs())

            label_pq[:] = 1
            pz = rbatch.data
            mod_dis.forward(mx.io.DataBatch(pz+rbatch.label, [label_pq]), is_train=True)
            mod_dis.backward()
            for gradsr, gradsf in zip(mod_dis._exec_group.grad_arrays, gradD):
                for gradr, gradf in zip(gradsr, gradsf):
                    gradr += gradf
            mod_dis.update()
            metric_dis_accuracy.update([label_pq], mod_dis.get_outputs())
            metric_dis_entropy.update([label_pq], mod_dis.get_outputs())

            #  step 2: update encoder(fool discriminator)
            label_pq[:] = 1
            mod_enc.forward(batch, is_train=True)
            qz = mod_enc.get_outputs()
            mod_dis.forward(mx.io.DataBatch(qz+[batch_label_one_hot], [label_pq]), is_train=True)
            mod_dis.backward()
            diff_dis = mod_dis.get_input_grads()
            mod_enc.backward([diff_dis[1]])
            mod_enc.update()

            #  metric update
            mod_dis.forward(mx.io.DataBatch(qz, [label_pq]), is_train=False)
            metric_fool_dis_accuracy.update([label_pq], mod_dis.get_outputs())
            metric_fool_dis_entropy.update([label_pq], mod_dis.get_outputs())

            if t % display_step == 0:
                print '\rEpoch %d, iter %d: dec_rmse=%.2f, dis_acc=%.4f, dis_entropy=%.2f, fool_dis_acc=%.4f, fool_dis_entropy=%.4f' % (epoch, t, metric_dec_rmse.get()[1], metric_dis_accuracy.get()[1], metric_dis_entropy.get()[1], metric_fool_dis_accuracy.get()[1], metric_fool_dis_entropy.get()[1]),
                sys.stdout.flush()

                metric_dec_rmse.reset()
                metric_dis_accuracy.reset()
                metric_dis_entropy.reset()
                metric_fool_dis_accuracy.reset()
                metric_fool_dis_entropy.reset()

        if (epoch+1) % check_point == 0:
            print 'Saving...'
            sym_dec.save('cache/models/%s_%s_dec_supervised.json'%(dataset, z_prior))
            mod_dec.save_params('cache/models/%s_%s_dec_supervised_%04d.params'%(dataset, z_prior, epoch))
            sym_enc.save('cache/models/%s_%s_enc_supervised.json'%(dataset, z_prior))
            mod_enc.save_params('cache/models/%s_%s_enc_supervised_%04d.params'%(dataset, z_prior, epoch))
            sym_dis.save('cache/models/%s_%s_dis_supervised.json'%(dataset, z_prior))
            mod_dis.save_params('cache/models/%s_%s_dis_supervised_%04d.params'%(dataset, z_prior, epoch))
