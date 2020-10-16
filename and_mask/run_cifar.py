# This code was adapted from pytorch ignite CIFAR-10 example
import argparse

import ignite
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
from ignite.contrib.engines import common
from ignite.contrib.handlers import TensorboardLogger, ProgressBar
from ignite.contrib.handlers.tensorboard_logger import OutputHandler
from ignite.engine import Events, Engine, create_supervised_evaluator
from ignite.handlers import global_step_from_engine
from ignite.metrics import Accuracy, Loss
from ignite.utils import convert_tensor
from torch.optim.lr_scheduler import MultiStepLR

import and_mask.and_mask_utils as and_mask_utils
from and_mask.utils.ignite_cifar10_utils import get_train_test_loaders, get_model
from and_mask.optimizers.adam_flexible_weight_decay import AdamFlexibleWeightDecay


def run(output_dir, config):
    device = "cuda"

    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    # Rescale batch_size and num_workers
    ngpus_per_node = 1
    batch_size = config['batch_size']
    num_workers = int((config['num_workers'] + ngpus_per_node - 1) / ngpus_per_node)

    (train_loader,
     test_loader,
     mislabeled_train_loader) = get_train_test_loaders(
        path=config['data_path'],
        batch_size=batch_size,
        num_workers=num_workers,
        random_seed=config['seed'],
        random_labels_fraction=config['random_labels_fraction'],
    )

    model = get_model(num_classes=10)
    model = model.to(device)

    optimizer = AdamFlexibleWeightDecay(model.parameters(),
                                        lr=config['init_lr'],
                                        weight_decay_order=config['weight_decay_order'],
                                        weight_decay=config['weight_decay'])

    criterion = nn.CrossEntropyLoss().to(device)

    le = len(train_loader)
    lr_scheduler = MultiStepLR(optimizer,
                               milestones=[le * config['epochs'] * 3 // 4],
                               gamma=0.1)

    def _prepare_batch(batch, device, non_blocking):
        x, y = batch
        return (convert_tensor(x, device=device, non_blocking=non_blocking),
                convert_tensor(y, device=device, non_blocking=non_blocking))

    def process_function(unused_engine, batch):
        x, y = _prepare_batch(batch, device=device, non_blocking=True)

        model.train()
        optimizer.zero_grad()

        y_pred = model(x)

        if config['agreement_threshold'] > 0.0:
            # The "batch_size" in this function refers to the batch size per env
            # Since we treat every example as one env, we should set the parameter
            # n_agreement_envs equal to batch size
            mean_loss, masks = and_mask_utils.get_grads(
                agreement_threshold=config['agreement_threshold'],
                batch_size=1,
                loss_fn=criterion,
                n_agreement_envs=config['batch_size'],
                params=optimizer.param_groups[0]['params'],
                output=y_pred,
                target=y,
                method=args.method,
                scale_grad_inverse_sparsity=config['scale_grad_inverse_sparsity'],
            )
        else:
            mean_loss = criterion(y_pred, y)
            mean_loss.backward()

        optimizer.step()

        return {}

    trainer = Engine(process_function)
    metric_names = []
    common.setup_common_training_handlers(trainer,
                                          output_path=output_dir, lr_scheduler=lr_scheduler,
                                          output_names=metric_names,
                                          with_pbar_on_iters=True,
                                          log_every_iters=10)

    tb_logger = TensorboardLogger(log_dir=output_dir)
    tb_logger.attach(trainer,
                     log_handler=OutputHandler(tag="train",
                                               metric_names=metric_names),
                     event_name=Events.ITERATION_COMPLETED)

    metrics = {
        "accuracy": Accuracy(),
        "loss": Loss(criterion)
    }

    test_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device, non_blocking=True)
    train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device, non_blocking=True)
    mislabeled_train_evaluator = create_supervised_evaluator(model, metrics=metrics, device=device, non_blocking=True)

    def run_validation(engine):
        torch.cuda.synchronize()
        train_evaluator.run(train_loader)
        if config['random_labels_fraction'] > 0.0:
            mislabeled_train_evaluator.run(mislabeled_train_loader)
        test_evaluator.run(test_loader)

    def flush_metrics(engine):
        tb_logger.writer.flush()

    trainer.add_event_handler(Events.EPOCH_STARTED(every=1), run_validation)
    trainer.add_event_handler(Events.COMPLETED, run_validation)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, flush_metrics)

    ProgressBar(persist=False, desc="Train evaluation").attach(train_evaluator)
    ProgressBar(persist=False, desc="Test evaluation").attach(test_evaluator)
    ProgressBar(persist=False, desc="Train (mislabeled portion) evaluation").attach(mislabeled_train_evaluator)

    tb_logger.attach(train_evaluator,
                     log_handler=OutputHandler(tag="train",
                                               metric_names=list(metrics.keys()),
                                               global_step_transform=global_step_from_engine(trainer)),
                     event_name=Events.COMPLETED)
    tb_logger.attach(test_evaluator,
                     log_handler=OutputHandler(tag="test",
                                               metric_names=list(metrics.keys()),
                                               global_step_transform=global_step_from_engine(trainer)),
                     event_name=Events.COMPLETED)
    tb_logger.attach(mislabeled_train_evaluator,
                     log_handler=OutputHandler(tag="train_wrong",
                                               metric_names=list(metrics.keys()),
                                               global_step_transform=global_step_from_engine(trainer)),
                     event_name=Events.COMPLETED)

    trainer_rng = np.random.RandomState()
    trainer.run(train_loader, max_epochs=config['epochs'],
                seed=trainer_rng.randint(2 ** 32))

    tb_logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--data_path', type=str, default="/tmp/cifar_dataset")
    parser.add_argument('--agreement_threshold', type=float, required=True)
    parser.add_argument('--weight_decay', type=float, required=True)
    parser.add_argument('--method', type=str, choices=['and_mask', 'geom_mean'], required=True)
    parser.add_argument('--scale_grad_inverse_sparsity', type=int, required=True)
    parser.add_argument('--init_lr', type=float, required=True)
    parser.add_argument('--random_labels_fraction', type=float, required=True)
    parser.add_argument('--weight_decay_order', type=str,
                        choices=['before', 'after'], default='before')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=80)
    parser.add_argument('--epochs', type=int, default=80)
    args = parser.parse_args()

    assert torch.cuda.is_available()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    config = vars(args)

    print("Train on CIFAR10")
    print("- PyTorch version: {}".format(torch.__version__))
    print("- Ignite version: {}".format(ignite.__version__))
    print("- CUDA version: {}".format(torch.version.cuda))

    print("\n")
    print("Configuration:")
    for key, value in config.items():
        print("\t{}: {}".format(key, value))
    print("\n")

    try:
        run(args.output_dir, config)
    except KeyboardInterrupt:
        print("Catched KeyboardInterrupt -> exit")
    except Exception as e:
        raise e
