"""
Train a diffusion model on images.
"""
import os
import torch
torch.set_num_threads(20)
import argparse
from torch.utils.data import DataLoader
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
from torchvision import transforms
import itertools

def custom_data_loader(data_folder, batch_size):
    to_pil = transforms.ToPILImage()
    pth_files = [f for f in os.listdir(data_folder) if f.endswith('.pth')]
    pth_files.sort()
    data_loader = DataLoader(pth_files, batch_size=batch_size, shuffle=False,num_workers=20)
    for batch in data_loader:
        data = [torch.load(os.path.join(data_folder, filename)) for filename in batch]
        data = torch.cat(data, dim=0)
        yield data

def load_data_adv(data_folder, batch_size):
    pth_files = [f for f in os.listdir(data_folder) if f.endswith('.pth')]
    pth_files.sort()
    data_loader = itertools.cycle(pth_files)
    while True:
        batch = list(itertools.islice(data_loader, batch_size))
        if not batch:
            break
        data = [torch.load(os.path.join(data_folder, filename)) for filename in batch]
        data = torch.cat(data, dim=0)
        yield data


def main():

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        for i in range(gpu_count):
            device_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {device_name}")
    args = create_argparser().parse_args()
    
    dist_util.setup_dist()
    logger.configure()
    
    print("device: ", dist_util.setup_dist())    

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.to(dist_util.dev())
    print(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    
    #data = custom_data_loader(args.data_dir, args.batch_size)
    #data_gt = custom_data_loader(args.data_gt_dir, args.batch_size)
    #data = load_data_adv(args.data_dir, args.batch_size)
    #data_gt = load_data_adv(args.data_gt_dir, args.batch_size)


    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=args.deterministic,  # deterministic if True, yield results in a deterministic order.
    )
    """
    data_gt = load_data(
        data_dir=args.data_gt_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
        deterministic=args.deterministic,  # deterministic if True, yield results in a deterministic order.
    )
    """
    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        loss_type=args.loss_type,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=50000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        deterministic=False,
        loss_type='',
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
