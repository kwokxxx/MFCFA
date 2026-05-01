"""Functions for training and running EF prediction."""

import math
import os
import time

import click
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
import torch
import torchvision
import tqdm
import scipy

import echonet



@click.command("video")
@click.option("--data_dir", type=click.Path(exists=True, file_okay=False), default=None)
@click.option("--output", type=click.Path(file_okay=False), default=None)
@click.option("--task", type=str, default=["EF","ESV","EDV"])
@click.option("--model_name", type=click.Choice(
    sorted(name for name in torchvision.models.video.__dict__
           if name.islower() and not name.startswith("__") and callable(torchvision.models.video.__dict__[name]))),
    default="r2plus1d_18")
@click.option("--pretrained/--random", default=True)
@click.option("--weights", type=click.Path(exists=True, dir_okay=False), default=None)
@click.option("--run_test/--skip_test", default=True)
@click.option("--num_epochs", type=int, default=45)
@click.option("--lr", type=float, default=1e-4)
@click.option("--weight_decay", type=float, default=1e-4)
@click.option("--lr_step_period", type=int, default=15)
@click.option("--frames", type=int, default=32)
@click.option("--period", type=int, default=2)
@click.option("--num_train_patients", type=int, default=None)
@click.option("--num_workers", type=int, default=4)
@click.option("--batch_size", type=int, default=20)
@click.option("--device", type=str, default=None)
@click.option("--seed", type=int, default=0)

def run(
    data_dir=None,
    output=None,
    task=["EF","ESV","EDV"],

    model_name="r2plus1d_18",
    pretrained=True,
    weights=None,

    run_test=True,
    num_epochs=45,
    lr=1e-4,
    weight_decay=1e-4,
    lr_step_period=15,
    frames=32,
    period=2,
    num_train_patients=None,
    num_workers=4,
    batch_size=15,
    device=None,
    seed=0,
):

    """Trains/tests EF prediction model.

    \b
    Args:
        data_dir (str, optional): Directory containing dataset. Defaults to
            `echonet.config.DATA_DIR`.
        output (str, optional): Directory to place outputs. Defaults to
            output/video/<model_name>_<pretrained/random>/.
        task (str, optional): Name of task to predict. Options are the headers
            of FileList.csv. Defaults to ``EF''.
        model_name (str, optional): Name of model. One of ``mc3_18'',
            ``r2plus1d_18'', or ``r3d_18''
            (options are torchvision.models.video.<model_name>)
            Defaults to ``r2plus1d_18''.
        pretrained (bool, optional): Whether to use pretrained weights for model
            Defaults to True.
        weights (str, optional): Path to checkpoint containing weights to
            initialize model. Defaults to None.
        run_test (bool, optional): Whether or not to run on test.
            Defaults to False.
        num_epochs (int, optional): Number of epochs during training.
            Defaults to 35.
        lr (float, optional): Learning rate for SGD
            Defaults to 1e-4.
        weight_decay (float, optional): Weight decay for SGD
            Defaults to 1e-4.
        lr_step_period (int or None, optional): Period of learning rate decay
            (learning rate is decayed by a multiplicative factor of 0.1)
            Defaults to 15.
        frames (int, optional): Number of frames to use in clip
            Defaults to 32.
        period (int, optional): Sampling period for frames
            Defaults to 2.
        n_train_patients (int or None, optional): Number of training patients
            for ablations. Defaults to all patients.
        num_workers (int, optional): Number of subprocesses to use for data
            loading. If 0, the data will be loaded in the main process.
            Defaults to 4.
        device (str or None, optional): Name of device to run on. Options from
            https://pytorch.org/docs/stable/tensor_attributes.html#torch.torch.device
            Defaults to ``cuda'' if available, and ``cpu'' otherwise.
        batch_size (int, optional): Number of samples to load per batch
            Defaults to 20.
        seed (int, optional): Seed for random number generator. Defaults to 0.
    """

    # Seed RNGs
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Set default output directory
    if output is None:
        output = os.path.join("output", "video", "{}_{}_{}_{}".format(model_name, frames, period, "pretrained" if pretrained else "random"))
    os.makedirs(output, exist_ok=True)

    # Set device for computations
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = torchvision.models.video.__dict__[model_name](pretrained=pretrained)

    model.fc = torch.nn.Linear(model.fc.in_features, 3)
    model.fc.bias.data.fill_(55.6)

    if device.type == "cuda":
        model = torch.nn.DataParallel(model)
    model.to(device)

    if weights is not None:
        checkpoint = torch.load(weights)
        model.load_state_dict(checkpoint['state_dict'])

    # Set up optimizer
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    if lr_step_period is None:
        lr_step_period = math.inf
    scheduler = torch.optim.lr_scheduler.StepLR(optim, lr_step_period)

    # Compute mean and std
    mean, std = echonet.utils.get_mean_and_std(echonet.datasets.Echo(root=data_dir, split="train"))
    kwargs = {"target_type": task,
              "mean": mean,
              "std": std,
              "length": frames,
              "period": period,
              }

    # Set up datasets and dataloaders
    dataset = {}
    dataset["train"] = echonet.datasets.Echo(root=data_dir, split="train", **kwargs, pad=12)
    if num_train_patients is not None and len(dataset["train"]) > num_train_patients:
        # Subsample patients (used for ablation experiment)
        indices = np.random.choice(len(dataset["train"]), num_train_patients, replace=False)
        dataset["train"] = torch.utils.data.Subset(dataset["train"], indices)
    dataset["val"] = echonet.datasets.Echo(root=data_dir, split="val", **kwargs)

    # Run training and testing loops
    with open(os.path.join(output, "log.csv"), "a") as f:
        epoch_resume = 0
        best_efLoss = float("inf")
        try:
            # Attempt to load checkpoint
            checkpoint = torch.load(os.path.join(output, "checkpoint.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            optim.load_state_dict(checkpoint['opt_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_dict'])
            epoch_resume = checkpoint["epoch"] + 1
            best_efLoss = checkpoint["best_ef_loss"]
            f.write("Resuming from epoch {}\n".format(epoch_resume))
        except FileNotFoundError:
            f.write("Starting run from scratch\n")

        for epoch in range(epoch_resume, num_epochs):
            print("Epoch #{}".format(epoch), flush=True)
            for phase in ['train', 'val']:
                start_time = time.time()
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)

                ds = dataset[phase]
                dataloader = torch.utils.data.DataLoader(
                    ds, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=(device.type == "cuda"), drop_last=(phase == "train"))

                ef_loss, y_ef, yhat_ef = echonet.utils.video.run_epoch(model, dataloader, phase == "train", optim, device)

                f.write("{},{},{},{},{},{},{},{},{}\n".format(epoch,
                                                  phase,
                                                  ef_loss,
                                                  sklearn.metrics.r2_score(y_ef, yhat_ef),
                                                  time.time() - start_time,
                                                  y_ef.size,
                                                  sum(torch.cuda.max_memory_allocated() for i in
                                                      range(torch.cuda.device_count())),
                                                  sum(torch.cuda.max_memory_reserved() for i in
                                                      range(torch.cuda.device_count())),
                                                  batch_size))
                f.flush()
            scheduler.step()

            # Save checkpoint
            save = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'period': period,
                'frames': frames,
                'best_ef_loss': best_efLoss,
                'ef_loss': ef_loss,
                'r2_ef': sklearn.metrics.r2_score(y_ef, yhat_ef),
                'opt_dict': optim.state_dict(),
                'scheduler_dict': scheduler.state_dict(),
            }
            torch.save(save, os.path.join(output, "checkpoint.pt"))
            if ef_loss < best_efLoss:
                torch.save(save, os.path.join(output, "best.pt"))
                best_efLoss = ef_loss

        # Load best weights
        if num_epochs != 0:
            checkpoint = torch.load(os.path.join(output, "best.pt"))
            model.load_state_dict(checkpoint['state_dict'])
            f.write("Best validation loss {} from epoch {}\n".format(checkpoint["ef_loss"], checkpoint["epoch"]))
            f.flush()


        if run_test:
            for split in ["val", "test"]:

                # Performance with test-time augmentation
                ds = echonet.datasets.Echo(root=data_dir, split=split, **kwargs, clips=48)
                dataloader = torch.utils.data.DataLoader(
                    ds, batch_size=1, num_workers=num_workers, shuffle=False, pin_memory=(device.type == "cuda"))
                ef_loss, y_ef, yhat_ef = echonet.utils.video.run_epoch(model, dataloader, False, None, device, save_all=True, block_size=batch_size)
                f.write("{} (all clips) R2:   {:.3f} ({:.3f} - {:.3f})\n".format(split, *echonet.utils.bootstrap(y_ef, np.array(list(map(lambda x: x.mean(), yhat_ef))), sklearn.metrics.r2_score)))
                f.write("{} (all clips) MAE:  {:.2f} ({:.2f} - {:.2f})\n".format(split, *echonet.utils.bootstrap(y_ef, np.array(list(map(lambda x: x.mean(), yhat_ef))), sklearn.metrics.mean_absolute_error)))
                f.write("{} (all clips) RMSE: {:.2f} ({:.2f} - {:.2f})\n".format(split, *tuple(map(math.sqrt, echonet.utils.bootstrap(y_ef, np.array(list(map(lambda x: x.mean(), yhat_ef))), sklearn.metrics.mean_squared_error)))))
                f.flush()

                # Write full performance to file
                with open(os.path.join(output, "{}_predictions.csv".format(split)), "w") as g:
                    for (filename, pred) in zip(ds.fnames, yhat_ef):
                        for (i, p) in enumerate(pred):
                            g.write("{},{},{:.4f}\n".format(filename, i, p))
                echonet.utils.latexify()
                yhat_ef = np.array(list(map(lambda x: x.mean(), yhat_ef)))

                # Calculating the Pearson Correlation Coefficient
                correlation_coefficient, _ = scipy.stats.pearsonr(y_ef, yhat_ef)

                # Plot actual and predicted EF
                fig = plt.figure(figsize=(3, 3))
                lower = min(y_ef.min(), yhat_ef.min())
                upper = max(y_ef.max(), yhat_ef.max())
                plt.scatter(y_ef, yhat_ef, color="k", s=1, edgecolor=None, zorder=2)
                plt.plot([0, 100], [0, 100], linewidth=1, zorder=3)
                plt.axis([lower - 3, upper + 3, lower - 3, upper + 3])
                plt.gca().set_aspect("equal", "box")
                plt.xlabel("Actual EF (%)")
                plt.ylabel("Predicted EF (%)")
                plt.xticks([10, 20, 30, 40, 50, 60, 70, 80])
                plt.yticks([10, 20, 30, 40, 50, 60, 70, 80])
                plt.grid(color="gainsboro", linestyle="--", linewidth=1, zorder=1)
                plt.text(0.5, 0.95, "Correlation coefficient = {:.2f}".format(correlation_coefficient),
                         horizontalalignment="center", verticalalignment="top",
                         transform=plt.gca().transAxes, color="red", fontsize=8)
                plt.tight_layout()
                plt.savefig(os.path.join(output, "{}_scatter.pdf".format(split)))
                plt.close(fig)

                # Plot AUROC
                fig = plt.figure(figsize=(3, 3))
                plt.plot([0, 1], [0, 1], linewidth=1, color="k", linestyle="--")
                for thresh in [35, 40, 45, 50]:
                    fpr, tpr, _ = sklearn.metrics.roc_curve(y_ef > thresh, yhat_ef)
                    print(thresh, sklearn.metrics.roc_auc_score(y_ef > thresh, yhat_ef))
                    plt.plot(fpr, tpr)

                plt.axis([-0.01, 1.01, -0.01, 1.01])
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.tight_layout()
                plt.savefig(os.path.join(output, "{}_roc.pdf".format(split)))
                plt.close(fig)



def run_epoch(model, dataloader, train, optim, device, save_all=False, block_size=None):
    """Run one epoch of training/evaluation for segmentation.

    Args:
        model (torch.nn.Module): Model to train/evaulate.
        dataloder (torch.utils.data.DataLoader): Dataloader for dataset.
        train (bool): Whether or not to train model.
        optim (torch.optim.Optimizer): Optimizer
        device (torch.device): Device to run on
        save_all (bool, optional): If True, return predictions for all
            test-time augmentations separately. If False, return only
            the mean prediction.
            Defaults to False.
        block_size (int or None, optional): Maximum number of augmentations
            to run on at the same time. Use to limit the amount of memory
            used. If None, always run on all augmentations simultaneously.
            Default is None.
    """

    model.train(train)

    ef_loss = 0
    esv_loss = 0
    edv_loss = 0
    loss = 0
    total = 0
    n = 0

    yhat_ef = []
    y_ef = []

    with torch.set_grad_enabled(train):
        with tqdm.tqdm(total=len(dataloader)) as pbar:
            for (X, label) in dataloader:
                y_ef.append(label["EF"].numpy())
                X = X.to(device)
                ef_label = label['EF'].to(device)
                esv_label = label['ESV'].to(device)
                edv_label = label['EDV'].to(device)

                average = (len(X.shape) == 6)

                if average:
                    batch, n_clips, c, f, h, w = X.shape
                    X = X.view(-1, c, f, h, w)


                if block_size is None:
                    pred = model(X)
                    ef_pred = pred[:,0]
                    esv_pred = pred[:,1]
                    edv_pred = pred[:,2]
                else:
                    pred = torch.cat(
                        [model(X[j:(j + block_size), ...]) for j in range(0, X.shape[0], block_size)])
                    ef_pred = pred[:, 0]
                    esv_pred = pred[:, 1]
                    edv_pred = pred[:, 2]

                if save_all:
                    yhat_ef.append(ef_pred.view(-1).to("cpu").detach().numpy())

                if average:
                    ef_pred = ef_pred.view(batch, n_clips, -1).mean(1)
                    esv_pred = esv_pred.view(batch, n_clips, -1).mean(1)
                    edv_pred = edv_pred.view(batch, n_clips, -1).mean(1)

                if not save_all:
                    yhat_ef.append(ef_pred.view(-1).to("cpu").detach().numpy())

                ef_loss = torch.nn.functional.mse_loss(ef_pred.view(-1), ef_label)
                esv_loss = torch.nn.functional.mse_loss(esv_pred.view(-1), esv_label)
                edv_loss = torch.nn.functional.mse_loss(edv_pred.view(-1), edv_label)
                loss = 0.9 * ef_loss + 0.09 * esv_loss + 0.01 * edv_loss

                if train:
                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                total += loss.item() * X.size(0)
                n += X.size(0)

                pbar.set_postfix_str("{:.2f}".format(total / n))
                pbar.update()

    if not save_all:
        yhat_ef = np.concatenate(yhat_ef)
    y_ef = np.concatenate(y_ef)

    return total / n, y_ef, yhat_ef
