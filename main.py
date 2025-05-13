import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, LearningRateFinder

from dataset import courtDataset # Assuming this remains the same
from court_lightning_module import CourtLitModule # Import the new Lightning Module
from batch_visualization_callback import SaveBatchCallback # Import the new callback

# Removed save_checkpoint function as PyTorch Lightning handles it via ModelCheckpoint

def main():
    # Add PyTorch 2.0 performance setting for Tensor Cores
    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high') 

    torch.multiprocessing.set_sharing_strategy('file_system')

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--exp_id', type=str, default='default', help='Experiment ID for saving results')
    parser.add_argument('--num_epochs', type=int, default=500, help='Total training epochs (max_epochs for Trainer)')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate for the model')
    parser.add_argument('--val_intervals', type=int, default=5, help='Epoch intervals to run validation (check_val_every_n_epoch)')
    # steps_per_epoch is now limit_train_batches if it's meant to limit training length per epoch
    parser.add_argument('--limit_train_batches', type=float, default=1.0, 
                        help='Percentage of training batches to run per epoch (1.0 for all, or an int for specific count)') 
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    # backup_path is not directly used by ModelCheckpoint in the same way, but we can save multiple checkpoints
    # We'll use exp_id to define checkpoint paths.
    # parser.add_argument('--backup_path', type=str, default=None, help='Path to save backup checkpoints') 
    # train_image_log_interval is omitted for now, can be added with a custom callback
    # parser.add_argument('--train_image_log_interval', type=int, default=100,
    #                     help='Iterations interval to log training images')
    
    # Add any other specific hparams for CourtLitModule if needed, e.g., val_max_dist
    parser.add_argument('--val_max_dist', type=int, default=3, help='Max distance for TP in validation')
    parser.add_argument('--num_keypoints', type=int, default=14, help='Number of keypoints')
    # Add output_width and output_height arguments for the LitModule
    parser.add_argument('--output_width', type=int, default=640, help='Width of the model output heatmap')
    parser.add_argument('--output_height', type=int, default=360, help='Height of the model output heatmap')
    parser.add_argument('--save_first_n_train_batches', type=int, default=3, # Reverted default to 0
                        help='Number of initial training batches to visualize and save (0 to disable)')

    args = parser.parse_args()

    # Prepare Datasets and Dataloaders (largely unchanged)
    train_dataset = courtDataset('train', augment=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_dataset = courtDataset('val') # Validation set, typically without augmentation
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Initialize Lightning Module
    lit_model = CourtLitModule(
        lr=args.lr,
        val_max_dist=args.val_max_dist,
        num_keypoints=args.num_keypoints,
        output_width=args.output_width,      # Pass to LitModule
        output_height=args.output_height,    # Pass to LitModule
    )

    # Paths for Logging and Checkpointing
    exps_path = os.path.join('exps', args.exp_id)
    
    # Logger
    tb_logger = TensorBoardLogger(save_dir=exps_path, name="plots", version="")

    # Checkpointing Callbacks
    checkpoint_callback_best = ModelCheckpoint(
        dirpath=os.path.join(exps_path, "checkpoints"),
        filename='model_best_{epoch}-{val_accuracy:.4f}',
        save_top_k=1,
        monitor='val_accuracy',
        mode='max',
    )

    # Callback to save a checkpoint every epoch
    checkpoint_callback_epoch = ModelCheckpoint(
        dirpath=os.path.join(exps_path, "checkpoints"),
        filename='model_epoch_{epoch}',
        save_top_k=-1,  # Save all checkpoints
        every_n_epochs=1 # Explicitly save every epoch
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    

    callbacks = [checkpoint_callback_best, checkpoint_callback_epoch, lr_monitor]

    # Add LearningRateFinder callback
    lr_finder = LearningRateFinder()
    callbacks.append(lr_finder)

    if args.save_first_n_train_batches > 0:
        vis_save_dir = os.path.join(exps_path, "batch_visualizations")
        save_batch_callback = SaveBatchCallback(
            save_dir=vis_save_dir,
            num_batches_to_save=args.save_first_n_train_batches
        )
        callbacks.append(save_batch_callback)

    # Determine accelerator and devices
    if torch.backends.mps.is_available():
        accelerator = "mps"
        devices = 1
    elif torch.cuda.is_available():
        accelerator = "gpu"
        devices = torch.cuda.device_count() if torch.cuda.device_count() > 0 else 1
    else:
        accelerator = "cpu"
        devices = 1
    print(f"Using accelerator: {accelerator}, devices: {devices}")

    # Handle checkpoint resumption logic before initializing Trainer if it's an old checkpoint
    ckpt_path_for_trainer = args.resume if args.resume and os.path.isfile(args.resume) else None
    
    if ckpt_path_for_trainer:
        print(f"Attempting to resume from checkpoint: {ckpt_path_for_trainer}")
        try:
            checkpoint = torch.load(ckpt_path_for_trainer, map_location=lambda storage, loc: storage) 
            
            if 'pytorch-lightning_version' in checkpoint:
                print("Checkpoint is in PyTorch Lightning format. Trainer will handle resumption.")
            elif 'model_state_dict' in checkpoint: # Check for old format
                print("Checkpoint is in old custom format. Manually loading model_state_dict.")
                lit_model.model.load_state_dict(checkpoint['model_state_dict'])
                # Optionally, handle 'epoch' and 'optimizer_state_dict' if needed for exact resume
                # old_epoch = checkpoint.get('epoch', -1)
                # print(f"Old checkpoint epoch: {old_epoch}. New training will start fresh unless epoch is passed to Trainer differently.")
                print(f"Manually loaded model weights from old checkpoint.")
                ckpt_path_for_trainer = None # Don't pass to Trainer, as we've loaded weights
            else: # Try loading as a raw state_dict if other checks fail
                print("Checkpoint format not recognized as Lightning or old custom. Attempting to load as raw model state_dict.")
                lit_model.model.load_state_dict(checkpoint)
                print("Manually loaded model weights from presumed raw state_dict.")
                ckpt_path_for_trainer = None
        except Exception as e:
            print(f"Error loading checkpoint {ckpt_path_for_trainer}: {e}. Will proceed without resuming from this checkpoint.")
            ckpt_path_for_trainer = None

    # Initialize Trainer
    trainer = pl.Trainer(
        logger=tb_logger,
        callbacks=callbacks,
        max_epochs=args.num_epochs,
        check_val_every_n_epoch=args.val_intervals,
        limit_train_batches=args.limit_train_batches if args.limit_train_batches <= 1.0 else int(args.limit_train_batches),
        accelerator=accelerator,
        devices=devices,
        gradient_clip_val=0.5,
        precision='16-mixed',
    )

    # Start Training
    print(f"Starting/Resuming training. Effective checkpoint path for trainer: {ckpt_path_for_trainer}")
    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path_for_trainer)

    # writer.close() # TensorBoardLogger handles this

if __name__ == '__main__':
    main()
