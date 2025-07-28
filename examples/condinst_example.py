#!/usr/bin/env python3

"""
CondInst Integration Example for nnDetection
============================================

This example demonstrates how to use the newly integrated CondInst (Conditional Instance Segmentation) 
functionality within the nnDetection framework.

Requirements:
- nnDetection environment set up
- Dataset prepared (e.g., Task025_LymphNodes)
- CondInst configuration enabled

Usage:
    python condinst_example.py --task 025 --fold 0
"""

import argparse
import os
from pathlib import Path

from nndet.training.trainer import run_training
from nndet.inference.predictor import run_prediction
from nndet.evaluator.detection import run_evaluation


def train_condinst_model(data_dir: str, task: str, fold: int):
    """
    Train CondInst model on specified task and fold
    
    Args:
        data_dir: path to preprocessed data directory
        task: task identifier (e.g., "025")
        fold: fold number (0-4)
    """
    print(f"Training CondInst model on Task{task}, Fold {fold}")
    
    # Set up environment variables
    os.environ['det_data'] = data_dir
    os.environ['det_models'] = str(Path(data_dir).parent / "models") 
    
    # Train with CondInst configuration
    run_training(
        task=f"Task{task}_CondInst",
        fold=fold,
        train_cfg="v001_condinst",  # Use our CondInst configuration
        augment_cfg="base_more",
        plan="D3V001_3d",
        trainer_cfg="v001",
        module="RetinaUNetV001",
        gpus=1,
        mixed_precision=True,
    )


def predict_with_condinst(data_dir: str, task: str, fold: int, model_dir: str):
    """
    Run prediction using trained CondInst model
    
    Args:
        data_dir: path to preprocessed data directory
        task: task identifier
        fold: fold number
        model_dir: path to trained model directory
    """
    print(f"Running prediction with CondInst model on Task{task}, Fold {fold}")
    
    run_prediction(
        task=f"Task{task}_CondInst",
        fold=fold,
        model_dir=model_dir,
        predictor="BoxPredictorSelective",
        save_seg=True,  # Enable instance mask saving
        save_raw_pred=True,
    )


def evaluate_condinst_results(data_dir: str, task: str, fold: int, pred_dir: str):
    """
    Evaluate CondInst detection and segmentation results
    
    Args:
        data_dir: path to preprocessed data directory
        task: task identifier
        fold: fold number  
        pred_dir: path to prediction results
    """
    print(f"Evaluating CondInst results for Task{task}, Fold {fold}")
    
    # Evaluate detection performance
    run_evaluation(
        task=f"Task{task}_CondInst",
        fold=fold,
        pred_dir=pred_dir,
        gt_dir=data_dir,
        evaluator="BoxEvaluator",
    )
    
    # TODO: Add instance segmentation evaluation when available
    print("Instance segmentation evaluation to be implemented")


def main():
    parser = argparse.ArgumentParser(description="CondInst Integration Example")
    parser.add_argument("--task", type=str, required=True, help="Task number (e.g., 025)")
    parser.add_argument("--fold", type=int, default=0, help="Fold number (0-4)")
    parser.add_argument("--data_dir", type=str, help="Path to preprocessed data")
    parser.add_argument("--mode", choices=["train", "predict", "evaluate", "all"], 
                       default="all", help="Operation mode")
    
    args = parser.parse_args()
    
    # Set default data directory if not provided
    if args.data_dir is None:
        args.data_dir = os.environ.get('det_data', f'/path/to/nnDetection_preprocessed/Task{args.task}_Dataset')
    
    data_dir = Path(args.data_dir)
    models_dir = data_dir.parent / "models"
    model_dir = models_dir / f"Task{args.task}_CondInst" / f"fold_{args.fold}"
    pred_dir = models_dir / "predictions" / f"Task{args.task}_CondInst" / f"fold_{args.fold}"
    
    print("=" * 60)
    print("CondInst Integration for nnDetection")
    print("=" * 60)
    print(f"Task: {args.task}")
    print(f"Fold: {args.fold}")
    print(f"Data directory: {data_dir}")
    print(f"Models directory: {models_dir}")
    print("=" * 60)
    
    if args.mode in ["train", "all"]:
        train_condinst_model(str(data_dir), args.task, args.fold)
    
    if args.mode in ["predict", "all"]:
        if model_dir.exists():
            predict_with_condinst(str(data_dir), args.task, args.fold, str(model_dir))
        else:
            print(f"Model directory {model_dir} not found. Skipping prediction.")
    
    if args.mode in ["evaluate", "all"]:
        if pred_dir.exists():
            evaluate_condinst_results(str(data_dir), args.task, args.fold, str(pred_dir))
        else:
            print(f"Prediction directory {pred_dir} not found. Skipping evaluation.")
    
    print("=" * 60)
    print("CondInst integration example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
