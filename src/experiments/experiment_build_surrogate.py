import os
from itertools import chain
from multiprocessing import Process
from pathlib import Path

import yaml
from pytorch_lightning.utilities.seed import seed_everything
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.experiments.experiment_problem import load_problem_data, problem_domains
from src.surrogate import SurrogateVAE, ZeroOneProblemData
from src.types_ import *


def train_task(problem_root_dir: str = "../../data/problem_instance", domain: str = None, train_dim: int = 80,
               train_num: int = 5, gpu_index: int = -1):
    device = torch.device("cuda") if gpu_index >= 0 else torch.device("cpu")
    with open("../../configs/surrogate.yaml", 'r') as file:
        config = yaml.safe_load(file)
    config["vae_params"]["in_dim"] = train_dim
    config["vae_params"]["latent_dim"] = train_dim * config["vae_params"]["latent_dim_coefficient"]
    config["vae_params"]["ins_emb_dim"] = train_dim * config["vae_params"]["ins_emb_dim_coefficient"]
    config["trainer_params"]["gpus"] = [gpu_index]
    config["logging_params"]["name"] = "{}-{}".format(domain, train_dim)
    seed_everything(config['exp_params']['manual_seed'], True)
    vae_model = SurrogateVAE(**config["vae_params"]).to(device)
    ins_embs: List[Tensor] = [torch.FloatTensor(config["vae_params"]["ins_emb_dim"]).to(device) for _ in
                              range(train_num)]
    train_dataloaders = []
    valid_dataloaders = []
    for ins_index in range(train_num):
        x, y = load_problem_data(
            problem_dir=Path(problem_root_dir, "train", "{}_{}_{}".format(domain, train_dim, ins_index))
        )
        train_data = ZeroOneProblemData(x, y, 'train')
        valid_data = ZeroOneProblemData(x, y, 'valid')
        train_dataloaders.append(
            DataLoader(train_data, batch_size=config['data_params']['train_batch_size'], shuffle=True,
                       num_workers=config['data_params']['num_workers']))
        valid_dataloaders.append(
            DataLoader(valid_data, batch_size=config['data_params']['val_batch_size'], shuffle=True,
                       num_workers=config['data_params']['num_workers']))
    log_path = Path(config['logging_params']['save_dir'], config["logging_params"]["name"])
    if not os.path.exists(Path(config['logging_params']['save_dir'])):
        os.mkdir(Path(config['logging_params']['save_dir']))
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    writer = SummaryWriter(str(log_path))
    yaml.dump(config, open(Path(log_path, "HyperParam.yaml"), "w"))
    for param in ins_embs:
        torch.nn.init.normal_(param, mean=0, std=1)
    all_parameters = [vae_model.parameters()] + [emb for emb in ins_embs]
    optimizer = optim.Adam(chain(*all_parameters),
                           lr=config['exp_params']['LR'],
                           weight_decay=config['exp_params']['weight_decay'])

    best_val_loss = np.inf
    epoch_bar = tqdm(range(int(config['trainer_params']['max_epochs'])))
    for epoch in epoch_bar:
        loss_records = {
            "Train": [],
            "Valid": [],
            "KLD": [],
            "KLD_VAL": [],
            "Scorer": [],
            "Scorer_VAL": [],
            "STD": [],
            "STD_VAL": [],
            "Reconstruction": [],
            "Reconstruction_VAL": []
        }
        for index in range(train_num):
            loss_records["Scorer_{}".format(index)] = []
            loss_records["Scorer_{}_VAL".format(index)] = []
        vae_model.train()
        for datas in zip(*train_dataloaders):
            optimizer.zero_grad()
            losses = []
            for train_index, (solution, quality) in enumerate(datas):
                solution, quality = solution.to(device), quality.to(device)
                ins_emb = ins_embs[train_index]
                forward_result = vae_model.forward(solution, ins_emb, train=True)
                loss = vae_model.loss(solution, quality, forward_result)
                current_loss = loss["Reconstruction_Loss"] / train_num + \
                               config["loss_params"]["gamma"] * loss["KLD_Loss"] + config["loss_params"]["lamBDa"] * \
                               loss["Performance_Loss"]
                current_loss.backward()
                losses.append({
                    "Reconstruction_Loss": loss["Reconstruction_Loss"].cpu().detach().numpy(),
                    "KLD_Loss": loss["KLD_Loss"].cpu().detach().numpy(),
                    "Performance_Loss": loss["Performance_Loss"].cpu().detach().numpy(),
                    "STD": torch.std(forward_result[2]).cpu().detach().numpy()
                })
            optimizer.step()
            recon_loss = np.mean([loss["Reconstruction_Loss"] for loss in losses])
            kld_loss = np.mean([loss["KLD_Loss"] for loss in losses])
            score_loss = np.mean([loss["Performance_Loss"] for loss in losses])
            recon_std = np.mean([loss["STD"] for loss in losses])
            train_loss = recon_loss + config["loss_params"]["gamma"] * kld_loss + config["loss_params"][
                "lamBDa"] * score_loss
            loss_records["Train"].append(train_loss)
            loss_records["KLD"].append(kld_loss)
            loss_records["Reconstruction"].append(recon_loss)
            loss_records["Scorer"].append(score_loss)
            loss_records["STD"].append(recon_std)
            for index in range(train_num):
                loss_records["Scorer_{}".format(index)].append(losses[index]["Performance_Loss"])
        vae_model.eval()
        for datas in zip(*valid_dataloaders):
            losses = []
            for train_index, (solution, quality) in enumerate(datas):
                solution, quality = solution.to(device), quality.to(device)
                ins_emb = ins_embs[train_index]
                forward_result = vae_model.forward(solution, ins_emb, train=False)
                loss = vae_model.loss(solution, quality, forward_result)
                losses.append({
                    "Reconstruction_Loss": loss["Reconstruction_Loss"].cpu().detach().numpy(),
                    "KLD_Loss": loss["KLD_Loss"].cpu().detach().numpy(),
                    "Performance_Loss": loss["Performance_Loss"].cpu().detach().numpy(),
                    "STD": torch.std(forward_result[2]).cpu().detach().numpy()
                })
            recon_loss = np.mean([loss["Reconstruction_Loss"] for loss in losses])
            kld_loss = np.mean([loss["KLD_Loss"] for loss in losses])
            score_loss = np.max([loss["Performance_Loss"] for loss in losses])
            recon_std = np.mean([loss["STD"] for loss in losses])
            valid_loss = recon_loss + config["loss_params"]["gamma"] * kld_loss + config["loss_params"][
                "lamBDa"] * np.mean([loss["Performance_Loss"] for loss in losses])
            loss_records["Valid"].append(valid_loss)
            loss_records["KLD_VAL"].append(kld_loss)
            loss_records["Reconstruction_VAL"].append(recon_loss)
            loss_records["Scorer_VAL"].append(score_loss)
            loss_records["STD_VAL"].append(recon_std)
            for index in range(train_num):
                loss_records["Scorer_{}_VAL".format(index)].append(losses[index]["Performance_Loss"])
        for key in loss_records.keys():
            writer.add_scalar(key, np.mean(loss_records[key]), epoch)
        epoch_bar.set_description("Epoch {}".format(epoch))
        epoch_bar.set_postfix_str("MSE {:.5f}".format(np.mean(loss_records['Train'])))
        if epoch >= 1000:
            if np.mean(loss_records['Valid']) < best_val_loss:
                best_val_loss = np.mean(loss_records['Valid'])
                save_dict = {"vae_model": vae_model.state_dict()}
                for index in range(train_num):
                    save_dict["instance_embedding_{}".format(index)] = ins_embs[index]
                print("Store {} in {} epoch".format(domain, epoch))
                torch.save(save_dict, Path(log_path, "best_model.pt"))
            elif np.mean(loss_records['Valid']) > best_val_loss * 5:
                print("Early Stop")
                break


def main():
    problem_root_dir = str(Path(os.path.dirname(os.path.abspath(__file__)), "../../data/problem_instance"))
    process_list = []
    useful_gpu = [5, 6, 7]
    task_index = 0
    for domain in problem_domains:
        for train_dim in problem_domains[domain]["train_dim"]:
            p = Process(target=train_task,
                        args=(problem_root_dir, domain, train_dim, problem_domains[domain]["train_num"],
                              useful_gpu[task_index % len(useful_gpu)]))
            p.start()
            process_list.append(p)
            task_index += 1
    for p in process_list:
        p.join()


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main()
