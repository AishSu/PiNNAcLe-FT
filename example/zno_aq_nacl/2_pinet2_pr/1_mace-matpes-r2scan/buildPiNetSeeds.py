import math
import os
import numpy as np
import argparse
import yaml
import tensorflow as tf
from pinn import get_model, get_network
from pinn.utils import init_params
from pinn.io import write_tfrecord, load_tfrecord, sparse_batch
from tempfile import mkdtemp, mkstemp

def buildSplitDataset(strWorkDir, strFileName, nSeed, nTrainRatio=8):
    """Handles generation of train/eval splits for a specific seed."""
    strDataFile = os.path.join(strWorkDir, "datasets", f"{strFileName}.yml")
    train_path = os.path.join(strWorkDir, f'train_seed{nSeed}.yml')
    eval_path = os.path.join(strWorkDir, f'eval_seed{nSeed}.yml')

    if not os.path.exists(train_path) or not os.path.exists(eval_path):
        print(f"Generating training/eval splits for seed {nSeed}...")
        dataset = load_tfrecord(strDataFile,
                                splits={'train': nTrainRatio, 'eval': 10 - nTrainRatio},
                                shuffle=True, seed=nSeed)
        write_tfrecord(train_path, dataset['train'])
        write_tfrecord(eval_path, dataset['eval'])
    return train_path, eval_path

def buildModel(nSeed):
    # Important parameters
    strFileName = "ZnO_aqNaCl_ZnCl2-330K-macer2scan-5854"
    nDepth, dRc, nGaussBasis, nModelSize = 5, 6.0, 10, 16
    dLearnRate, dLearnRateDecay = 5.0e-05, 0.994
    nTrainSteps = 5000000
    nEvalSteps = 50
    nBatch = 1

    # Other parameters
    nlog_every = 5000
    nckpt_every = 5000
    nmax_ckpts = 1
    nshuffleBuffer = 1000
    bPreprocess = True
    bCache = True
    bEarlyStop = False

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    strWorkDir = os.getcwd() + "/"

    # 1. Dataset Splitting
    train_path, eval_path = buildSplitDataset(strWorkDir, strFileName, nSeed)

    # 2. Model Parameters
    params = {}
    params["model"] = { 
        "name": "potential_model",
        "params": {  
            "use_force": True,
            "e_loss_multiplier": 10.0,
            "f_loss_multiplier": 100.0,
            "e_scale": 1.0,
            "e_unit": 1.0,
            "use_e_per_atom": False,
            "log_e_per_atom": True,
        }
    }

    params["network"] = {   
        "name": "PiNet2",
        "params": {   
            "atom_types": [1, 8, 30, 17, 11],
            "depth": nDepth,
            "rc": dRc,
            "n_basis": nGaussBasis,
            "basis_type": "gaussian",
            "pi_nodes": [nModelSize],
            "pp_nodes": [nModelSize]*4,
            "ii_nodes": [nModelSize]*4,
            "out_nodes": [nModelSize],
            "rank": 3,
            "weighted": False
        }
    }

    params["optimizer"] = {     
        "class_name": "Adam",
        "config": {
            "global_clipnorm": 0.01,
            "learning_rate": {
                "class_name": "ExponentialDecay",
                "config": {
                    "decay_rate": dLearnRateDecay,
                    "decay_steps": 100000,
                    "initial_learning_rate": dLearnRate,
                }
            }
        }
    }

    # 3. Log Parameters
    model_params_path = os.path.join(strWorkDir, f'pinet2_seed{nSeed}.yml')
    if not os.path.exists(model_params_path):
        with open(model_params_path, 'w') as f:
            yaml.dump(params, f)

    # 4. Model Directory
    exp = int(np.floor(np.log10(nTrainSteps)))
    steps_str = f"{int(nTrainSteps/(10**exp))}E{exp}"
    params['model_dir'] = f"{strFileName}-pinet2-B{nBatch}-{steps_str}-{nSeed}"

    # 5. Data Pipeline
    ds = load_tfrecord(train_path)
    init_params(params, ds)

    def _dataset_fn(fname):
        dataset = load_tfrecord(fname)
        if nBatch is not None:
            dataset = dataset.apply(sparse_batch(nBatch))
        if bPreprocess:
            dataset = dataset.map(lambda t: get_network(params['network']).preprocess(t),
                                  num_parallel_calls=tf.data.AUTOTUNE)
        if bCache:
            dataset = dataset.cache('')
        return dataset.prefetch(tf.data.AUTOTUNE)

    train_fn = lambda: _dataset_fn(train_path).repeat().shuffle(nshuffleBuffer)
    eval_fn = lambda: _dataset_fn(eval_path)

    config = tf.estimator.RunConfig(
        keep_checkpoint_max=nmax_ckpts,
        log_step_count_steps=nlog_every,
        save_checkpoints_steps=nckpt_every
    )

    model = get_model(params, config=config)
    train_spec = tf.estimator.TrainSpec(input_fn=train_fn, max_steps=nTrainSteps)
    eval_spec  = tf.estimator.EvalSpec(input_fn=eval_fn, steps=nEvalSteps)
    
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    buildModel(args.seed)
