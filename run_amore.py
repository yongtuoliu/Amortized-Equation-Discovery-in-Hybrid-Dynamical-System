from cgi import test
import os
import json
import cv2
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import argparse
import numpy as np
import matplotlib
from tensorboardX import SummaryWriter
import src.utils as utils
import src.datasets as datasets
import src.tensorboard_utils as tensorboard_utils
from src.model_utils import build_model
from src.evaluation import evaluate_segmentation
from src.torch_utils import torch2numpy
import random
import matplotlib.pyplot as plt
from sklearn.metrics.cluster import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    contingency_matrix,
)
from scipy.optimize import linear_sum_assignment

available_datasets = {"spring-mass"}

def train_step(batch, model, optimizer, step, config):
    model.train()

    def _set_lr(lr):
        for g in optimizer.param_groups:
            g["lr"] = lr

    # Increasing temperature increase the uncertainty about each discrete states
    # when step<5000, switch_temp equals 100, after 5000, decay gradually to 10
    switch_temp = utils.get_temperature(step, config, "switch_")

    extra_args = dict()
    # when step<5000, switch_temp equals 10, after 5000, decay gradually to 1
    dur_temp = utils.get_temperature(step, config, "dur_")
    extra_args = {"dur_temperature": dur_temp}
    
    cont_ent_anneal = config["cont_ent_anneal"] # 1.0
    # when step<2000, xent equals 100, after 2000, decay gradually to 1e-10
    xent_coeff = utils.get_cross_entropy_coef(step, config)
    # when step<2000, lr increases from 5.e-5 to 0.0004, after 2000, decay gradually
    lr = utils.get_learning_rate(step, config)

    optimizer.zero_grad()
    result = model(
        batch,
        switch_temperature=switch_temp,
        num_samples=config["num_samples"],
        cont_ent_anneal=cont_ent_anneal,
        **extra_args,
    )
    objective = -1 * (
        result[config["objective"]] + xent_coeff * result["crossent_regularizer"]
    ) # config["objective"]: elbov2; 
    result["objective"] = objective
    result["lr"] = lr
    result["switch_temperature"] = switch_temp
    result["dur_temperature"] = dur_temp
    result["cont_ent"] = cont_ent_anneal
    result["xent_coeff"] = xent_coeff
    objective.backward()
    nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip_norm"])
    _set_lr(lr)
    optimizer.step()
    return result


def plot_results(result, n_obj, prefix=""):
    
    for i in range(int(n_obj)):
        original_inputs = torch2numpy(result["inputs"][i])
        reconstructed_inputs = torch2numpy(result["reconstructions"][i])
        most_likely_states = torch2numpy(torch.argmax(result["log_gamma"], dim=-1)[i])
        hidden_states = torch2numpy(result["x_samples"][i])
        discrete_states_lk = torch2numpy(torch.exp(result["log_gamma"])[i])
        true_seg = None
        if "true_seg" in result:
            true_seg = torch2numpy(result["true_seg"][i])
            true_seg = np.asarray(true_seg, dtype=np.int8)

        ylim = 1.3 * np.abs(original_inputs).max()
        matplotlib_fig = tensorboard_utils.show_time_series(
            fig_size=(12, 4),
            inputs=original_inputs,
            reconstructed_inputs=reconstructed_inputs,
            segmentation=most_likely_states,
            true_segmentation=true_seg,
            fig_title="input_reconstruction",
            ylim=(-ylim, ylim),
        )
        fig_numpy_array = tensorboard_utils.plot_to_image(matplotlib_fig)
        summary.add_image(
            "Object_{}_Reconstruction".format(str(i)), fig_numpy_array, step, dataformats="HWC"
        )
    matplotlib_fig = tensorboard_utils.show_hidden_states(
        fig_size=(12, 3), zt=hidden_states, segmentation=most_likely_states
    )
    fig_numpy_array = tensorboard_utils.plot_to_image(matplotlib_fig)
    summary.add_image(
        f"{prefix}Hidden_State_xt", fig_numpy_array, step, dataformats="HWC"
    )
    matplotlib_fig = tensorboard_utils.show_discrete_states(
        fig_size=(12, 3),
        discrete_states_lk=discrete_states_lk,
        segmentation=most_likely_states,
    )
    fig_numpy_array = tensorboard_utils.plot_to_image(matplotlib_fig)
    summary.add_image(
        f"{prefix}Discrete_State_zt", fig_numpy_array, step, dataformats="HWC"
    )


def get_dataset(config):
    dataset = config["dataset"]
    assert dataset in available_datasets, f"Unknown dataset {dataset}!"
    if dataset == "spring-mass":
        train_dataset = datasets.SpringMassDataset(path="./data/10012023Hopping_dataset.mat", data_type="train", n_obj=config['n_obj'])
        val_dataset = datasets.SpringMassDataset(path="./data/10012023Hopping_dataset.mat", data_type="val", n_obj=config['n_obj'])
        test_dataset = datasets.SpringMassDataset(path="./data/10012023Hopping_dataset.mat", data_type="test", n_obj=config['n_obj'])
    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    matplotlib.use("Agg")

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # COMMAND-LINE ARGS
    parser = argparse.ArgumentParser()
    # group = parser.add_mutually_exclusive_group(required=True)
    # group.add_argument("--config", type=str, help="Path to config file.")
    # group.add_argument("--ckpt", type=str, help="Path to checkpoint file.")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/spring-mass.yaml",
        help="Path to config file.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Which device to use, e.g., cpu, cuda:0, cuda:1, ...",
    )
    args = parser.parse_args()

    # CONFIG
    # if args.ckpt:
    #     ckpt = torch.load(args.ckpt, map_location="cpu")
    #     config = ckpt["config"]
    # else:
    #     config = utils.get_config_and_setup_dirs(args.config)
    config = utils.get_config_and_setup_dirs(args.config)
    device = torch.device(args.device)
    with open(os.path.join(config["log_dir"], "config.json"), "w") as fp:
        json.dump(config, fp)

    # DATA
    train_dataset, val_dataset, test_dataset = get_dataset(config)
    # # Check data
    # print(train_dataset.data_aggre[0,0,130,3], 11-10*train_dataset.data_aggre[0,0,130,0], train_dataset.data_aggre[0,0,130,4])
    # import pdb; pdb.set_trace()
    
    num_workers = 0
    def _init_fn(worker_id):
        np.random.seed(seed + worker_id)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"], # 40
        num_workers=num_workers, # 0
        shuffle=True,
        pin_memory=True,
        worker_init_fn=_init_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=config["batch_size"], pin_memory=True, worker_init_fn=_init_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config["batch_size"], pin_memory=True, worker_init_fn=_init_fn
    )

    train_gen = iter(train_loader)
    val_gen = iter(val_loader)
    test_gen = iter(test_loader)

    print(f'Running {config["model"]} on {config["dataset"]}.')
    print(f"Train size: {len(train_dataset)}. Val size: {len(val_dataset)}. Test size: {len(test_dataset)}.")
    # import pdb; pdb.set_trace()

    # MODEL
    model = build_model(config=config)
    start_step = 0
    # if args.ckpt:
    #     model.load_state_dict(ckpt["model"])
    #     start_step = ckpt["step"] + 1
    model = model.to(device)

    # TRAIN AND EVALUATE
    optimizer = torch.optim.Adam(
        model.parameters(), weight_decay=config["weight_decay"]
    )
    # if args.ckpt:
    #     optimizer.load_state_dict(ckpt["optimizer"])

    summary = SummaryWriter(logdir=config["log_dir"])
    f1_score_best = 0
    accuracy_score_best = 0
    best_model_path_list = []
    eval_seg_metrics = open(os.path.join(config["log_dir"], "eval_seg_metrics.txt"), "w")
    eval_seg_metrics_best = open(os.path.join(config["log_dir"], "eval_seg_metrics_best.txt"), "w")
    for step in range(start_step, config["num_steps"]):
        try:
            train_batch_with_label = next(train_gen) # (B, num_o, num_f, 5)
            # print(train_batch_with_label.shape, train_batch_sign.shape, train_batch_edge.shape)
            num_b, num_o, num_f, num_d = train_batch_with_label.shape
            train_batch_with_label = train_batch_with_label.view(-1, num_f, num_d)
            train_batch = train_batch_with_label[:,:,:2] # (B,T,(x,y)) # [B, 150, 2]
            train_label = train_batch_with_label[:,:,-1] # # [B, 150]
            train_batch = train_batch.to(device)
        except StopIteration:
            train_gen = iter(train_loader)
        train_result = train_step(train_batch, model, optimizer, step, config)

        if step % config["log_steps"] == 0 or step == config["num_steps"]:
            train_result["true_seg"] = train_label
            plot_results(train_result, config["n_obj"])

            # Plot duration models
            dummy_ctrls = torch.ones(1, 1, 1, device=device)
            rho = torch2numpy(
                model.ctrl2nstf_network.rho(
                    dummy_ctrls, temperature=train_result["dur_temperature"]
                )
            )[0, 0]
            matplotlib_fig = tensorboard_utils.show_duration_dists(
                fig_size=(15, rho.shape[0] * 2), rho=rho
            )
            fig_numpy_array = tensorboard_utils.plot_to_image(matplotlib_fig)
            summary.add_image("Duration", fig_numpy_array, step, dataformats="HWC")

            # Evaluate Segmentation
            print('Evaluation...')
            extra_args = dict()
            extra_args = {"dur_temperature": train_result["dur_temperature"]}
            true_segs = []
            pred_segs = []
            true_tss = []
            recons_tss = []
            for val_batch_with_label in val_loader:
                num_b, num_o, num_f, num_d = val_batch_with_label.shape
                val_batch_with_label = val_batch_with_label.view(-1, num_f, num_d)
                val_batch = val_batch_with_label[:,:,:2] # [B,T,2]
                val_label = val_batch_with_label[:,:,-1] # # [B, T]
                val_batch = val_batch.to(device)
                val_result = model(
                    val_batch,
                    switch_temperature=train_result["switch_temperature"],
                    num_samples=1,
                    deterministic_inference=True,
                    **extra_args,
                )
                pred_seg = torch2numpy(torch.argmax(val_result["log_gamma"], dim=-1)) # (B,T), B: Batchsize and n_obj
                true_seg = torch2numpy(val_label) # (B,T)
                true_seg = np.asarray(true_seg, dtype=np.int8) # [B, T]
                true_ts = torch2numpy(val_result["inputs"]) # (B,T,2)
                recons_ts = torch2numpy(val_result["reconstructions"]) # (B,T,2)
                true_tss.append(true_ts)
                recons_tss.append(recons_ts)
                true_segs.append(true_seg)
                pred_segs.append(pred_seg)
            true_tss = np.concatenate(true_tss, 0)
            recons_tss = np.concatenate(recons_tss, 0)
            true_segs = np.concatenate(true_segs, 0)
            pred_segs = np.concatenate(pred_segs, 0)
            seg_metrics = evaluate_segmentation(true_segs, pred_segs, K=config["num_categories"])
            print('step: {}, seg_metrics: {}'.format(step, seg_metrics))
            eval_seg_metrics.write('step: {}, metrics: {}\r'.format(step, seg_metrics))
            eval_seg_metrics.flush()
            f1_score_now = seg_metrics['f1_score']
            accuracy_score_now = seg_metrics['accuracy']
            if f1_score_now > f1_score_best and accuracy_score_now > accuracy_score_best:
                best_model_path = os.path.join(config["model_best_dir"], f"model_{step}.pt")
                torch.save(
                    {
                        "step": step,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": config,
                        'swith-temp': train_result["switch_temperature"],
                        'dur-temp': train_result["dur_temperature"],
                    },
                    best_model_path,
                )
                best_model_path_list.append(best_model_path)
                f1_score_best = f1_score_now
                accuracy_score_best = accuracy_score_now
                eval_seg_metrics_best.write('step: {}, metrics: {}\r'.format(step, seg_metrics))
                eval_seg_metrics_best.flush()

            summary_items = {
                "params/learning_rate": train_result["lr"],
                "params/switch_temperature": train_result["switch_temperature"],
                "params/dur_temperature": train_result["dur_temperature"],
                "params/cross_entropy_coef": train_result["xent_coeff"],
                "elbo/training": train_result[config["objective"]],
                "xent/training": train_result["crossent_regularizer"],
            }
            for k, v in seg_metrics.items():
                summary_items[f"metrics/{k}"] = v
            for k, v in summary_items.items():
                summary.add_scalar(k, v, step)
            summary.flush()

    # visualizing test set Segmentation
    if True:
        best_model_path = best_model_path_list[-1]
        ckpt = torch.load(best_model_path, map_location="cpu")
        model = model.to(torch.device('cpu'))
        model.load_state_dict(ckpt["model"])
        model = model.to(device)

        best_switch_temp = ckpt['swith-temp']
        best_dur_temp = ckpt['dur-temp']
        extra_args = dict()
        extra_args = {"dur_temperature": best_dur_temp}
        
        original_inputs = []
        reconstructed_inputs = []
        true_segs = []
        most_likely_states = []
        for test_batch_with_label in test_loader:
            num_b, num_o, num_f, num_d = test_batch_with_label.shape
            test_batch_with_label = test_batch_with_label.view(-1, num_f, num_d)
            test_batch = test_batch_with_label[:,:,:2] 
            test_label = test_batch_with_label[:,:,-1]
            test_batch = test_batch.to(device)
            test_result = model(
                test_batch,
                switch_temperature=best_switch_temp,
                num_samples=1,
                deterministic_inference=True,
                **extra_args,
            )
            original_input = torch2numpy(test_result["inputs"])
            reconstructed_input = torch2numpy(test_result["reconstructions"])
            true_seg = np.asarray(torch2numpy(test_label), dtype=np.int8)
            most_likely_state = torch2numpy(torch.argmax(test_result["log_gamma"], dim=-1))
            original_inputs.append(original_input)
            reconstructed_inputs.append(reconstructed_input)
            true_segs.append(true_seg)
            most_likely_states.append(most_likely_state)
        original_inputs = np.concatenate(original_inputs, 0)
        reconstructed_inputs = np.concatenate(reconstructed_inputs, 0)
        true_segs = np.concatenate(true_segs, 0)
        most_likely_states = np.concatenate(most_likely_states, 0)
        # match segmentation ids
        true_segs_for_ids = true_segs
        pred_segs_for_ids = most_likely_states
        true_segs_for_ids = true_segs_for_ids.reshape(-1)
        pred_segs_for_ids = pred_segs_for_ids.reshape(-1)
        unique_labels_for_ids = np.unique(true_segs_for_ids)
        unique_preds_for_ids = np.unique(pred_segs_for_ids)
        cont_mat = contingency_matrix(true_segs_for_ids, pred_segs_for_ids)
        cost_matrix = cont_mat.max() - cont_mat
        row_idx, col_idx = linear_sum_assignment(cost_matrix)
        mapper = {unique_preds_for_ids[c]: unique_labels_for_ids[r] for (r, c) in zip(row_idx, col_idx)}
        for i in unique_preds_for_ids:
            if i not in mapper:
                mapper[i] = 1000
        print(mapper)
        most_likely_states_shape = most_likely_states.shape
        most_likely_states_for_ids = np.reshape(most_likely_states, -1)
        
        remapped_most_likely_states_for_ids = np.array([mapper[i] for i in most_likely_states_for_ids])
        remapped_most_likely_states = np.reshape(remapped_most_likely_states_for_ids, most_likely_states_shape)
        most_likely_states = remapped_most_likely_states

        original_inputs = np.reshape(original_inputs, (-1,config["n_obj"],150,2))
        reconstructed_inputs = np.reshape(reconstructed_inputs, (-1,config["n_obj"],150,2))
        true_segs = np.reshape(true_segs, (-1,config["n_obj"],150))
        most_likely_states = np.reshape(most_likely_states, (-1,config["n_obj"],150))
        n_batch, n_obj, n_frame, _ = original_inputs.shape
        for i in range(n_batch):
            for j in range(n_obj):
                each_input = original_inputs[i][j]
                each_recon = reconstructed_inputs[i][j]
                each_seg = true_segs[i][j]
                each_sta = most_likely_states[i][j]
                ylim = 1.3 * np.abs(each_input).max()
                matplotlib_fig = tensorboard_utils.show_time_series(
                    fig_size=(12, 4),
                    inputs=each_input,
                    reconstructed_inputs=each_recon,
                    segmentation=each_sta,
                    true_segmentation=each_seg,
                    fig_title="input_reconstruction_{}".format(str(j)),
                    ylim=(-ylim, ylim),
                )
                plt.savefig(config["log_dir"]+"{}.png".format(str(j)))
                plt.close()
            all_images = []
            for j in range(n_obj):
                each_image = cv2.imread(config["log_dir"]+"{}.png".format(str(j)))
                all_images.append(each_image)
            hei, wid, _ = each_image.shape
            new_image = np.zeros((len(all_images)*hei, wid, 3))
            for j in range(len(all_images)):
                new_image[j*hei:(j+1)*hei,:,:] = all_images[j]
            cv2.imwrite(config["log_dir"]+"{}_whole.png".format(str(i)), new_image)
            for j in range(n_obj):
                os.remove(config["log_dir"]+"{}.png".format(str(j)))

    # Evaluating test setSegmentation
    test_seg_metrics = open(os.path.join(config["log_dir"], "test_seg_metrics.txt"), "w")
    if True:
        for each_best_model_path in best_model_path_list:
            ckpt = torch.load(each_best_model_path, map_location="cpu")
            model = model.to(torch.device('cpu'))
            model.load_state_dict(ckpt["model"])
            model = model.to(device)
            step = ckpt["step"]

            best_switch_temp = ckpt['swith-temp']
            best_dur_temp = ckpt['dur-temp']
            extra_args = dict()
            extra_args = {"dur_temperature": best_dur_temp}

            true_segs = []
            pred_segs = []
            true_tss = []
            recons_tss = []
            for test_batch_with_label in test_loader:
                num_b, num_o, num_f, num_d = test_batch_with_label.shape
                test_batch_with_label = test_batch_with_label.view(-1, num_f, num_d)
                test_batch = test_batch_with_label[:,:,:2] # (B,T,(x,y)) # [40, 100, 2]
                test_label = test_batch_with_label[:,:,-1] # # [40, 100]
                test_batch = test_batch.to(device)
                test_result = model(
                    test_batch,
                    switch_temperature=best_switch_temp,
                    num_samples=1,
                    deterministic_inference=True,
                    **extra_args,
                )
                pred_seg = torch2numpy(
                    torch.argmax(test_result["log_gamma"], dim=-1)
                )
                true_seg = torch2numpy(test_label[:, :config["context_length"]])
                true_seg = np.asarray(true_seg, dtype=np.int8)
                true_ts = torch2numpy(test_result["inputs"])
                recons_ts = torch2numpy(test_result["reconstructions"])
                true_tss.append(true_ts)
                recons_tss.append(recons_ts)
                true_segs.append(true_seg)
                pred_segs.append(pred_seg)
            true_tss = np.concatenate(true_tss, 0)
            recons_tss = np.concatenate(recons_tss, 0)
            true_segs = np.concatenate(true_segs, 0)
            pred_segs = np.concatenate(pred_segs, 0)
            true_tss = true_tss
            recons_tss = recons_tss
            true_segs = true_segs
            pred_segs = pred_segs
            seg_metrics = evaluate_segmentation(
                true_segs, pred_segs, K=config["num_categories"]
            )
            print('step: {}, test_seg_metrics: {}'.format(step, seg_metrics))
            test_seg_metrics.write('step: {}, metrics: {}\r'.format(step, seg_metrics))
            test_seg_metrics.flush()

    # Delete abundant best models, only three left
    if True:
        for each_best_model_path in best_model_path_list[:-3]:
            os.remove(each_best_model_path)