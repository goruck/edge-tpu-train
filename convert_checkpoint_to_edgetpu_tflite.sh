#!/bin/bash
# Converts TensorFlow checkpoint to EdgeTPU-compatible TFLite file.
# Copyright (c) 2019 Lindo St. Angel.

# Exit script on error.
set -e
# Echo each command, easier for debugging.
set -x

usage() {
  cat << END_OF_USAGE
  Converts TensorFlow checkpoint to EdgeTPU-compatible TFLite file.

  --pipeline_config_path - Path to pipeline config file (default .configs/radar-ml/pipeline_mobilenet_v2_ssd_retrain_last_few_layers.config).
  --train_dir - Path to train directory (default ./train).
  --checkpoint_num - Checkpoint number (default 0).
  --output_dir - Output directory (default ./)
  --help - Display this help.
END_OF_USAGE
}

INPUT_TENSORS='normalized_input_image_tensor'
OUTPUT_TENSORS='TFLite_Detection_PostProcess,TFLite_Detection_PostProcess:1,TFLite_Detection_PostProcess:2,TFLite_Detection_PostProcess:3'

# Defaults - will get overridden if provided on cmd line.
pipeline_config_path=./configs/radar-ml/pipeline_mobilenet_v2_ssd_retrain_last_few_layers.config
train_dir=./train
ckpt_number=0
output_dir=./tflite_models/radar-ml

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pipeline_config_path)
      pipeline_config_path=$2
      shift 2 ;;
    --train_dir)
      train_dir=$2
      shift 2 ;;
    --checkpoint_num)
      ckpt_number=$2
      shift 2 ;;
    --output_dir)
      output_dir=$2
      shift 2 ;;
    --help)
      usage
      exit 0 ;;
    --*)
      echo "Unknown flag $1"
      usage
      exit 1 ;;
  esac
done

echo "EXPORTING frozen graph from checkpoint..."
python3 ~/develop/tensorflow/models/research/object_detection/export_tflite_ssd_graph.py \
  --pipeline_config_path="${pipeline_config_path}" \
  --trained_checkpoint_prefix="${train_dir}/model.ckpt-${ckpt_number}" \
  --output_directory="${output_dir}" \
  --add_postprocessing_op=true
  echo "Frozen graph generated at ${output_dir}/tflite_graph.pb"

echo "CONVERTING frozen graph to TF Lite file..."
tflite_convert \
  --output_file="${output_dir}/output_tflite_graph.tflite" \
  --graph_def_file="${output_dir}/tflite_graph.pb" \
  --inference_type=QUANTIZED_UINT8 \
  --input_arrays="${INPUT_TENSORS}" \
  --output_arrays="${OUTPUT_TENSORS}" \
  --mean_values=128 \
  --std_dev_values=128 \
  --input_shapes=1,300,300,3 \
  --change_concat_input_ranges=false \
  --allow_nudging_weights_to_use_fast_gemm_kernel=true \
  --allow_custom_ops
echo "TFLite graph generated at ${output_dir}/output_tflite_graph.tflite"

echo "COMPILING TFLite graph for edge tpu..."
edgetpu_compiler \
  --out_dir ${output_dir} \
  ${output_dir}/output_tflite_graph.tflite
echo "Edge tpu graph generated at ${output_dir}/output_tflite_graph_edgetpu.tflite"