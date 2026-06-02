<div align="center">

# RetCond: A Conditional Diffusion Model for Self-Explanatory Multi-Class Fundus Image Classification

</div>
This repository contains the code associtaed with the paper:
> O Ahsan, A., Nielsen, C., Souza, R., Aulakh, A., D Forkert, N., & Wilms, M. (2026). RetCond: A Conditional Diffusion Model for Self-Explanatory Multi-Class Fundus Image Classification. Journal of Medical Systems, 50(1), 84.

[[Link to paper]](https://link.springer.com/article/10.1007/s10916-026-02414-0)

## Setup Environment

```bash
python3 -m venv generative_discriminative
source generative_discriminative/bin/activate
pip3 install -r requirements.txt
```

## Setup Data

Create the dataset in the ImageFolder structure:

```
data/
├── Train/
│   ├── AMD/
│   ├── DR/
│   └── Normal/
├── Validation/
│   ├── AMD/
│   ├── DR/
│   └── Normal/
└── Test/
    ├── AMD/
    ├── DR/
    └── Normal/
```

## Configuration

All hyperparameters and paths are configured in YAML files (e.g., `conf.yaml`). Key configuration sections:

### `exp` section (Experiment Settings)
- `exp_name`: Experiment identifier
- `training_type`: Type of training to run (see Training Modes below)
- `data_dir`: Path to training data
- `ckpt_dir`: Directory to save model checkpoints
- `logdir`: TensorBoard logs directory (default: `logs/`)
- `csv_dir`: Directory for output CSV results
- `model_ckpt_path`: Path to load checkpoint from (for continue training)
- `device`: `auto` or specific GPU IDs
- `accelerator`: `auto`, `gpu`, or `cpu`
- `num_workers`: Number of data loading workers
- Other paths: `pickle_dir`, `sample_image_dir`, `counterfactual_dir`, `predict_dir`

### `hparams` section (Hyperparameters)
- `batch_size`: Training batch size (default: 64)
- `min_epochs`, `max_epochs`: Training duration
- `num_classes`: Number of classification classes (default: 6)
- `learning_rate`: Learning rate (default: 0.0001)
- `seed`: Random seed for reproducibility
- `image_h`, `image_w`: Image dimensions (default: 224x224)
- `DiffusionModelUnet`: Architecture configuration for diffusion model
  - `num_res_blocks`: Residual blocks per level
  - `num_channels`: Channel dimensions per level
  - `attention_levels`: Where to apply attention
  - `num_head_channels`: Attention head channels
  - `with_conditioning`: Enable class conditioning
  - `num_class_embeds`: Number of classes for conditioning
- `DDPMScheduler`/`DDIMScheduler`: Noise scheduling configuration
- `transform`: Data augmentation settings (color jitter, auto augment, etc.)

## Training Modes

The training pipeline supports multiple training types configured via `exp.training_type`:

### Diffusion Models (UKBB Dataset)
- `scratch_UKBB`: Train DDPM from scratch on UK BioBank data
- `continue_UKBB`: Continue training DDPM from a checkpoint

### Conditional Diffusion Models
- `scratch_conditional`: Train conditional DDIM from scratch on retinal data
- `scratch_SLO_conditional`: Train conditional DDIM on SLO (Scanning Laser Ophthalmoscopy) data
- `scratch_SLO_test`: Test conditional DDIM on SLO data
- `pretrained`: Fine-tune pretrained DDPM
- `continue_pretrained`: Continue fine-tuning from checkpoint
- `continue_pretrained_classification`: Continue training for classification
- `cond_diff_continue`: Continue conditional diffusion training
- `predict`: Run inference on new data
- `test`: Evaluate on test set

### Classification Models
- `mlp`: Train simple MLP classifier
- `mlp_continue`: Continue MLP training
- `resnet_train`: Train ResNet-50 classifier
- `resnet_test`: Test ResNet-50
- `resnet_predict`: Run ResNet-50 inference
- `efficient_net_b3_train`: Train EfficientNet-B3
- `efficient_net_b3_test`: Test EfficientNet-B3
- `efficient_net_b3_predict`: Run EfficientNet-B3 inference
- `efficient_net_b0_train`: Train EfficientNet-B0
- `efficient_net_b0_test`: Test EfficientNet-B0
- `efficient_net_b0_predict`: Run EfficientNet-B0 inference
- `swin_b_train`: Train Swin Transformer-B
- `swin_b_retrain`: Retrain Swin Transformer-B from checkpoint
- `swin_b_test`: Test Swin Transformer-B
- `swin_b_predict`: Run Swin Transformer-B inference

## Code Structure

### `run.py` (Main Training Script)

The entry point for all training and evaluation pipelines. Uses PyTorch Lightning for training orchestration.

#### Key Components:
- **Helper Functions** (for clean, DRY code):
  - `create_checkpoint_callback(config, filename_prefix)`: Creates ModelCheckpoint with configurable filename prefix (diffusion, resnet, efficient_net_b3, swin_b, mlp)
  - `create_trainer(config, logger, callbacks)`: Instantiates PyTorch Lightning Trainer with standard configuration
  - `pipeline(config)`: Main orchestration function that routes to different training modes based on `exp.training_type`

- **Data Modules**:
  - `UK_biobank_data_module`: For UKBB diffusion training
  - `Retinal_Cond_Lightning_Split`: For conditional diffusion on retinal image data
  - `SLO_Lightning_Split`: For SLO (Scanning Laser Ophthalmoscopy) data
  - `Pickle_Lightning`: For MLP classifier training with pickled embeddings
  - `Camera_Lightning_Split`: For camera/imaging data
  - `Resampling_Lightning_Split`: For resampled datasets

- **Models**:
  - `LightningDDPM_monai`: Unconditional DDPM
  - `Pretrained_LightningDDPM_monai`: Fine-tuned DDPM with class conditioning
  - `Conditional_DDIM_monai`: Conditional DDIM for guided sampling
  - `MLP_classifier`: Simple multilayer perceptron for classification
  - `Resnet_50`: ResNet-50 classifier
  - `EfficientNet_B0/B3`: EfficientNet variants
  - `Swin_B`: Swin Transformer-B

### Other Scripts

- `utils.py`: Utility functions and data module implementations
- `model_architecture.py`: Model definitions and PyTorch Lightning wrappers
- `ckpt_loader_saver.py`: Checkpoint management utilities

## Usage

### Run Training

```bash
python run.py --conf conf.yaml
```

The script reads configuration from the specified YAML file and executes the pipeline based on `exp.training_type`.

Example with custom config:
```bash
python run.py --conf sample_config/resnet_train.yaml
```

### Monitor Training

```bash
tensorboard --logdir=logs --bind_all
```

View training metrics in TensorBoard at `http://localhost:6006`

Metrics tracked:
- Training/validation loss
- Learning rate changes
- Test accuracies
- Image samples (for diffusion models)

## Utility Scripts

### Generate Conditional Samples (sampler.py)

Generate diverse samples from a trained conditional diffusion model across all disease classes.

```bash
python sampler.py --conf conf.yaml
```

**What it does:**
- Loads a trained conditional DDPM/DDIM model
- Generates samples for each disease class (AMD, Cataract, DR, Glaucoma, Myopia, Normal)
- Uses DDIM scheduler for faster sampling
- Saves grid images showing one sample per class

**Configuration:**
- `model_ckpt_path`: Path to trained model checkpoint
- `sample_image_dir`: Output directory for generated images
- `number_of_samples`: Number of samples to generate
- `num_inference_timesteps`: Sampling steps (default: 500)

**Output:**
- Grid images saved as PNG files (e.g., `AMD_Cat_D_Gl_Mya_N_0.png`)
- Each image shows 6 class variations generated from same noise

### Add Noise to Images (noiser.py)

Add controlled noise to images following the diffusion noise schedule.

```bash
python noiser.py
```

**What it does:**
- Loads a single image or batch of images
- Applies noise at specified diffusion timesteps
- Uses the DDPM scheduler to add noise progressively
- Visualizes the noising process as a grid

**Configuration (hardcoded in script):**
- Input image path
- Timesteps to generate (default: [0] for baseline)
- Output directory: `Sample_noised/`

**Use cases:**
- Visualizing the diffusion process
- Creating progressive noise levels for analysis
- Preparing inputs for denoising studies

**Example timestep values:**
```python
timesteps = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
```

### Generate Counterfactual Explanations (generate_counterfactuals.py)

Generate counterfactual images to explain classifier predictions.

```bash
python generate_counterfactuals.py --conf conf.yaml
```

**What it does:**
- Loads a trained conditional diffusion model
- For each misclassified sample, generates a counterfactual image
- Creates samples that would change the prediction to the correct class
- Uses intermediate diffusion steps (noising) for better quality

**Configuration:**
- `model_ckpt_path`: Trained conditional DDPM checkpoint
- `trial_pickle_file`: Results pickle from classification model
- `counterfactual_dir`: Output directory for counterfactual images
- `denoising_timestep`: How deep in noise space to start (default: 996)
- `num_inference_timesteps`: Denoising sampling steps

**Output Structure:**
```
counterfactuals/
├── A_0_P_1_image1.png  # Actual: AMD, Predicted: Cataract
├── A_1_P_2_image2.png  # Actual: Cataract, Predicted: DR
└── ...
```

Filename format: `A_{actual_class}_P_{predicted_class}_{image_id}.png`

**Interpretation:**
- Shows what the image should look like to be classified as the true class
- Highlights discriminative features for each disease
- Provides human-interpretable explanations of model decisions

**Processing:**
- Loads ground truth labels from pickle file
- Batch processes images for efficiency
- Unnormalizes images using ImageNet statistics
- Saves high-quality PNG outputs

### Denoise Images (denoiser.py)

Denoise images using a trained diffusion model.

```bash
python denoiser.py --conf conf.yaml
```

**What it does:**
- Loads a trained diffusion model
- Applies reverse diffusion process to denoise noisy inputs
- Can be used for image enhancement or artifact removal

**Output:**
- Denoised images saved to configured directory

## Checkpoints

Model checkpoints are saved to the directory specified in `exp.ckpt_dir`. The checkpoint filename includes:
- Model type/prefix
- Epoch number
- Training step  
- Validation loss

Example filenames:
- `diffusion-epoch=200-step=50000-val_loss=0.00125.ckpt`
- `resnet-epoch=257-step=7998-val_loss=0.29630.ckpt`
- `swin_b-epoch=150-step=30000-val_loss=0.12345.ckpt`

Top-k checkpoints are kept based on validation loss (configured by `exp.save_top_k`). The last checkpoint is also saved for recovery.

## Output Structure

- `checkpoints/`: Model checkpoints organized by experiment
- `logs/`: TensorBoard logs for each experiment
- `scores/`: Evaluation metrics and predictions
- `Sampled_images/`: Generated samples from diffusion models
- `counterfactuals/`: Counterfactual samples for interpretability
- `difference_image/`: Image difference visualizations

## Target Classes

Default target disease classes for classification:
- AMD (Age-related Macular Degeneration)
- Cataract
- DR (Diabetic Retinopathy)
- Glaucoma
- Myopia
- Normal

Configurable via `hparams.target_names` in config file.

## Example Configurations

See `sample_config/` directory for example configurations for different training scenarios:
- Diffusion model training
- Classification model training
- Fine-tuning from pretrained models
- Evaluation/testing
- Prediction on new data


## Citation

If you would like to use our code or paper for your work please cite us.

```
@article{o2026retcond,
  title={RetCond: A Conditional Diffusion Model for Self-Explanatory Multi-Class Fundus Image Classification},
  author={O Ahsan, Ahmad and Nielsen, Christopher and Souza, Raissa and Aulakh, Agampreet and D Forkert, Nils and Wilms, Matthias},
  journal={Journal of Medical Systems},
  volume={50},
  number={1},
  pages={84},
  year={2026},
  publisher={Springer}
}

```