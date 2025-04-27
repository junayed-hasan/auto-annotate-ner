#!/bin/bash
# Shell script to run the complete annotation and evaluation pipeline

# Default values
MODEL="gpt-4o-mini"
BATCH_SIZE=10
MAX_SAMPLES=10   # Default to 10 samples for quick testing
DATASET="valid"  # Options: train, valid, test

# Print header
echo "============================================================"
echo "  Bangla HealthNER LLM Annotation Pipeline"
echo "============================================================"

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --max-samples)
      MAX_SAMPLES="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --full)
      # Process all samples
      MAX_SAMPLES=""
      shift
      ;;
    --help)
      echo "Usage: $0 [options]"
      echo "Options:"
      echo "  --model MODEL         LLM model to use (default: gpt-4o-mini)"
      echo "  --batch-size SIZE     Batch size for processing (default: 10)"
      echo "  --max-samples N       Maximum number of samples to process (default: 10)"
      echo "  --dataset DATASET     Dataset to use: train, valid, test (default: valid)"
      echo "  --full                Process all samples (overrides --max-samples)"
      echo "  --help                Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Print configuration
echo
echo "Configuration:"
echo "  Model:       $MODEL"
echo "  Batch size:  $BATCH_SIZE"
if [[ -z "$MAX_SAMPLES" ]]; then
  echo "  Samples:     ALL (full dataset)"
else
  echo "  Samples:     $MAX_SAMPLES"
fi
echo "  Dataset:     $DATASET"
echo

# Set file paths
INPUT_FILE="data/${DATASET}.json"
OUTPUT_FILE="data/${DATASET}_annotated.json"
EVAL_OUTPUT="data/${DATASET}_evaluation.json"
CONFUSION_MATRIX="data/${DATASET}_confusion_matrix.png"

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
  echo "Error: Input file $INPUT_FILE does not exist."
  exit 1
fi

# Create output directory if it doesn't exist
mkdir -p data

# Prepare command args
MAX_SAMPLES_ARG=""
if [[ ! -z "$MAX_SAMPLES" ]]; then
  MAX_SAMPLES_ARG="--max-samples $MAX_SAMPLES"
fi

# Step 1: Run annotation
echo "Step 1: Running annotation on $INPUT_FILE..."
python3 src/annotate.py --input "$INPUT_FILE" --output "$OUTPUT_FILE" --model "$MODEL" --batch-size "$BATCH_SIZE" $MAX_SAMPLES_ARG

# Check if annotation was successful
if [ ! -f "$OUTPUT_FILE" ]; then
  echo "Error: Annotation failed. Output file $OUTPUT_FILE not created."
  exit 1
fi

# Step 2: Run evaluation
echo
echo "Step 2: Evaluating annotations..."
python3 src/evaluate.py --gold "$INPUT_FILE" --pred "$OUTPUT_FILE" --output "$EVAL_OUTPUT" --plot "$CONFUSION_MATRIX" $MAX_SAMPLES_ARG

# Print completion message
echo
echo "Pipeline complete!"
echo "  ✓ Annotated data saved to: $OUTPUT_FILE"
echo "  ✓ Evaluation results saved to: $EVAL_OUTPUT"
echo "  ✓ Confusion matrix saved to: $CONFUSION_MATRIX"
echo
echo "To run on the full dataset, use: ./run_pipeline.sh --full --dataset $DATASET"
echo "To view detailed results: cat $EVAL_OUTPUT" 