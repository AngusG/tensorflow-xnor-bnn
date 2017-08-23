import os.path
import numpy as np

def create_dir_if_not_exists(dir):
    if not os.path.exists(dir):
        dir += '/1'
        os.makedirs(dir)
    else:
        sub_dirs = next(os.walk(dir))[1]
        if len(sub_dirs) > 0:
            print(dir)
            arr = np.asarray(sub_dirs).astype('int')
            sub = str(arr.max() + 1)
            print(sub)
            dir += '/' + sub
            print(dir)
        else:
            dir += '/1'
        os.makedirs(dir)
    print('Logging to %s' % dir)
    return dir

def handle_args(args):

    binary = first = last = xnor = batch_norm = False
    log_path = ''

    # handle command line args
    if args.binary:
        
        print("Using 1-bit weights and activations")
        binary = True
        
        # only binarize last layer if received binary flag
        if args.last:
            last = True
        if args.first:
            first = True
        
        if first and last:
            sub_1 = '/bin_all/'
        elif first and not last:
            sub_1 = '/bin_first/'
        elif last and not first:
            sub_1 = '/bin_last/'
        else:
            sub_1 = '/bin/'
        
        # only use xnor kernel if received binary flag
        if args.xnor:

            print("Using xnor xnor_gemm kernel")
            xnor = True
            sub_2 = 'xnor/'
        else:
            sub_2 = 'matmul/'
    else:
        sub_1 = '/fp/'
        sub_2 = ''

    if args.log_dir:
        log_path = args.log_dir + sub_1 + sub_2 + \
            'hid_' + str(args.n_hidden) + '/'

    if args.batch_norm:
        print("Using batch normalization")
        batch_norm = True
        if args.log_dir:
            log_path += 'batch_norm/'

    if args.log_dir:
        log_path += 'bs_' + str(args.batch_size) + '/keep_' + \
            str(args.keep_prob) + '/lr_' + str(args.lr)
        
        # reg is a bnn regularization only, while we do dropout for both bin and fp
        if binary:
            log_path += '/reg_' + str(args.reg)
        if args.extra:
            log_path += '/' + args.extra
        log_path = create_dir_if_not_exists(log_path)

    return log_path, binary, first, last, xnor, batch_norm