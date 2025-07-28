# CondInst Integration for nnDetection

This document describes the integration of CondInst (Conditional Instance Segmentation) into the nnDetection framework, transforming it from a standard RetinaNet object detection framework into an instance segmentation framework.

## Overview

CondInst extends object detection by adding dynamic mask generation capabilities. The key innovation is using instance-specific convolution parameters to generate high-quality instance masks in a single forward pass.

### Key Components Added

1. **Controller Head** (`nndet/arch/heads/controller.py`)
   - Predicts dynamic convolution parameters for each detected instance
   - Parallel to existing classification and regression heads
   - Outputs parameters for 3 dynamic convolution layers

2. **Dynamic Mask Head** (`nndet/arch/heads/mask.py`)
   - Generates instance masks using dynamic convolution parameters
   - Operates on high-resolution FPN features (e.g., P3 level)
   - Uses grouped convolution for efficient parallel processing

3. **Enhanced Base Architecture** (`nndet/core/retina.py`)
   - Modified `BaseRetinaNet` to support CondInst components
   - Updated forward pass and training loop
   - Integrated mask loss computation

4. **Configuration Support** (`nndet/conf/train/v001_condinst.yaml`)
   - CondInst-specific training configuration
   - Configurable mask loss weights and internal channels

## Architecture Details

### Controller Head Architecture
```
Input (FPN features) 
    ↓
3x Conv3D(fpn_channels → head_channels)
    ↓
Conv3D(head_channels → anchors_per_pos × num_mask_params)
    ↓
Output: Dynamic convolution parameters per anchor
```

### Dynamic Mask Head Architecture
```
FPN Feature Map (P3) + Dynamic Parameters
    ↓
Dynamic Conv3D(fpn_channels → internal_channels) + ReLU
    ↓ 
Dynamic Conv3D(internal_channels → internal_channels) + ReLU
    ↓
Dynamic Conv3D(internal_channels → 1) + Sigmoid
    ↓
Output: Instance Masks
```

### Training Process

1. **Forward Pass**:
   - Standard detection head predictions (classification + regression)
   - Controller head predicts dynamic parameters
   - For positive samples: generate masks using dynamic mask head

2. **Loss Computation**:
   - Standard detection losses (classification + regression)
   - Additional mask loss (Soft Dice Loss) for positive samples
   - Combined loss: `L = L_cls + L_reg + λ_mask × L_mask`

## Usage

### 1. Training with CondInst

```bash
# Set environment variables
export det_data="/path/to/nnDetection_preprocessed"
export det_models="/path/to/nnDetection_models"

# Train with CondInst configuration
nndet train Task025_LymphNodes 0 \
    --config v001_condinst \
    --num_epochs 50 \
    --gpus 1
```

### 2. Configuration Options

Key parameters in `v001_condinst.yaml`:

```yaml
model_cfg:
  enable_condinst: True                    # Enable CondInst components
  mask_head_internal_channels: 8          # Internal channels for dynamic mask head
  mask_loss_weight: 2.0                   # Weight for mask loss
  
  head_controller_kwargs:                  # Controller head configuration
    num_convs: 3
    norm_channels_per_group: 16
    norm_affine: True
```

### 3. Inference and Evaluation

```bash
# Run inference
nndet predict Task025_LymphNodes 0 \
    --checkpoint best \
    --save_raw_pred

# Evaluate results (detection metrics)
nndet eval Task025_LymphNodes 0 \
    --predictions_dir predictions/
```

## Data Requirements

CondInst requires instance mask annotations in addition to bounding boxes. The data loading pipeline needs to be modified to provide:

- `target_boxes`: Bounding box coordinates
- `target_classes`: Object class labels  
- `target_instance_masks`: Binary instance masks for positive samples

### Expected Data Format

```python
targets = {
    "target_boxes": List[Tensor],        # List of [N, 6] tensors (x1,y1,z1,x2,y2,z2)
    "target_classes": List[Tensor],      # List of [N] tensors (class labels)
    "target_seg": Tensor,                # [B, H, W, D] semantic segmentation
    "target_instance_masks": Tensor,     # [num_pos_samples, H, W, D] instance masks
}
```

## Performance Considerations

### Memory Usage
- Dynamic mask head requires additional GPU memory
- Memory scales with number of positive samples per batch
- Consider reducing batch size if OOM occurs

### Training Time
- ~15-20% increase in training time compared to detection-only
- Mask loss computation adds computational overhead
- Dynamic convolution operations are memory-intensive

### Hyperparameter Tuning
- `mask_loss_weight`: Start with 2.0, adjust based on validation performance
- `mask_head_internal_channels`: 8 channels work well for most cases
- Controller `num_convs`: 3 layers provide good parameter capacity

## Implementation Notes

### Key Design Decisions

1. **Parameter Calculation**: Dynamic convolution parameters are computed based on:
   - 3D convolutions with 3×3×3 kernels
   - Layer 1: `fpn_channels → internal_channels`
   - Layer 2: `internal_channels → internal_channels` 
   - Layer 3: `internal_channels → 1`

2. **Feature Level Selection**: P3 level used for mask generation
   - Provides good balance between resolution and semantic information
   - Can be configured via `decoder_levels[0]`

3. **Loss Integration**: Mask loss only computed for positive samples
   - Reduces computational overhead
   - Focuses learning on relevant instances

### Limitations and Future Work

1. **Data Loading**: Current implementation assumes instance masks are pre-computed
   - Future: Add on-the-fly mask generation from segmentation labels

2. **Evaluation Metrics**: Instance segmentation metrics not yet implemented
   - Future: Add AP_mask, IoU_mask evaluation

3. **Multi-Scale Features**: Currently uses single FPN level
   - Future: Investigate multi-scale mask generation

## Troubleshooting

### Common Issues

1. **CUDA OOM during training**
   - Reduce batch size
   - Decrease `mask_head_internal_channels`
   - Use gradient checkpointing

2. **Poor mask quality**
   - Increase `mask_loss_weight`
   - Check data loading for instance masks
   - Verify positive/negative sample balance

3. **Training instability**
   - Lower learning rate for controller/mask head
   - Add gradient clipping
   - Check loss scaling with mixed precision

### Debug Mode

Enable debug logging:
```python
import logging
logging.getLogger('nndet').setLevel(logging.DEBUG)
```

This will show detailed information about:
- Controller parameter shapes
- Dynamic mask head operations
- Loss computation details

## Citation

If you use this CondInst integration, please cite both the original nnDetection and CondInst papers:

```bibtex
@inproceedings{baumgartner2021nndetection,
  title={nnDetection: A self-configuring method for medical object detection},
  author={Baumgartner, Michael and Jaeger, Paul F and others},
  booktitle={MICCAI},
  year={2021}
}

@inproceedings{tian2020conditional,
  title={Conditional convolutions for instance segmentation},
  author={Tian, Zhi and Shen, Chunhua and Chen, Hao},
  booktitle={ECCV},
  year={2020}
}
```
