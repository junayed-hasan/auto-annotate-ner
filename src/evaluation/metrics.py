from typing import List, Dict, Any, Tuple, Set
from collections import Counter
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support


def ensure_string_labels(labels: List[Any]) -> List[str]:
    """
    Ensure all labels are strings.
    
    Args:
        labels: List of labels that might contain non-string values
        
    Returns:
        List of string labels
    """
    return [str(label) if not isinstance(label, str) else label for label in labels]


def compute_token_level_metrics(true_labels: List[Any], predicted_labels: List[Any]) -> Dict[str, float]:
    """
    Compute token-level metrics (accuracy, macro precision, recall, F1).
    
    Args:
        true_labels: List of true labels
        predicted_labels: List of predicted labels
        
    Returns:
        Dictionary with token-level metrics
    """
    # Convert labels to strings if needed
    true_labels_str = ensure_string_labels(true_labels)
    predicted_labels_str = ensure_string_labels(predicted_labels)
    
    # Check if lengths match
    if len(true_labels_str) != len(predicted_labels_str):
        raise ValueError(f"Lengths of true_labels ({len(true_labels_str)}) and predicted_labels ({len(predicted_labels_str)}) don't match.")
    
    # Token-level metrics
    accuracy = accuracy_score(true_labels_str, predicted_labels_str)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels_str, predicted_labels_str, average='macro')
    
    return {
        'token_accuracy': accuracy,
        'token_precision': precision,
        'token_recall': recall,
        'token_f1': f1
    }


def extract_entities(labels: List[Any]) -> List[Tuple[str, int, int]]:
    """
    Extract entity spans from a sequence of IOB labels.
    
    Args:
        labels: List of IOB labels
        
    Returns:
        List of (entity_type, start_idx, end_idx) tuples
    """
    # Convert labels to strings if needed
    labels = ensure_string_labels(labels)
    
    entities = []
    current_entity = None
    
    for i, label in enumerate(labels):
        if label == 'O':
            if current_entity:
                entity_type, start = current_entity
                entities.append((entity_type, start, i - 1))
                current_entity = None
        elif label.startswith('B-'):
            if current_entity:
                entity_type, start = current_entity
                entities.append((entity_type, start, i - 1))
            
            entity_type = label[2:]  # Remove "B-" prefix
            current_entity = (entity_type, i)
        elif label.startswith('I-'):
            if not current_entity:
                # This is an I- tag without a preceding B- tag, which is not valid IOB
                # We'll treat it as a B- tag for evaluation purposes
                entity_type = label[2:]  # Remove "I-" prefix
                current_entity = (entity_type, i)
    
    # Add the last entity if there is one
    if current_entity:
        entity_type, start = current_entity
        entities.append((entity_type, start, len(labels) - 1))
    
    return entities


def compute_entity_level_metrics(true_labels: List[Any], predicted_labels: List[Any]) -> Dict[str, float]:
    """
    Compute entity-level metrics (precision, recall, F1).
    
    Args:
        true_labels: List of true IOB labels
        predicted_labels: List of predicted IOB labels
        
    Returns:
        Dictionary with entity-level metrics
    """
    true_entities = set(extract_entities(true_labels))
    pred_entities = set(extract_entities(predicted_labels))
    
    # Entity-level metrics
    tp = len(true_entities & pred_entities)
    fp = len(pred_entities - true_entities)
    fn = len(true_entities - pred_entities)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'entity_precision': precision,
        'entity_recall': recall,
        'entity_f1': f1,
        'entity_tp': tp,
        'entity_fp': fp,
        'entity_fn': fn
    }


def compute_entity_type_metrics(true_labels: List[Any], predicted_labels: List[Any]) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for each entity type.
    
    Args:
        true_labels: List of true IOB labels
        predicted_labels: List of predicted IOB labels
        
    Returns:
        Dictionary with metrics for each entity type
    """
    true_entities = extract_entities(true_labels)
    pred_entities = extract_entities(predicted_labels)
    
    # Group by entity type
    true_by_type = {}
    for entity in true_entities:
        entity_type, start, end = entity
        if entity_type not in true_by_type:
            true_by_type[entity_type] = set()
        true_by_type[entity_type].add((start, end))
    
    pred_by_type = {}
    for entity in pred_entities:
        entity_type, start, end = entity
        if entity_type not in pred_by_type:
            pred_by_type[entity_type] = set()
        pred_by_type[entity_type].add((start, end))
    
    # Compute metrics for each type
    result = {}
    all_types = set(true_by_type.keys()) | set(pred_by_type.keys())
    
    for entity_type in all_types:
        true_spans = true_by_type.get(entity_type, set())
        pred_spans = pred_by_type.get(entity_type, set())
        
        tp = len(true_spans & pred_spans)
        fp = len(pred_spans - true_spans)
        fn = len(true_spans - pred_spans)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        result[entity_type] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': len(true_spans)
        }
    
    return result


def evaluate_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """
    Evaluate a single annotated sample.
    
    Args:
        sample: Sample dictionary with 'labels' and 'predicted_labels' keys
        
    Returns:
        Dictionary with evaluation metrics
    """
    true_labels = sample['labels']
    predicted_labels = sample['predicted_labels']
    
    # Check if we have predictions
    if not predicted_labels:
        return {
            'token_metrics': {
                'token_accuracy': 0.0,
                'token_precision': 0.0,
                'token_recall': 0.0,
                'token_f1': 0.0
            },
            'entity_metrics': {
                'entity_precision': 0.0,
                'entity_recall': 0.0,
                'entity_f1': 0.0,
                'entity_tp': 0,
                'entity_fp': 0,
                'entity_fn': 0
            },
            'error': 'No predictions'
        }
    
    # Check for length mismatch
    if len(true_labels) != len(predicted_labels):
        print(f"Warning: Length mismatch between true labels ({len(true_labels)}) and predicted labels ({len(predicted_labels)})")
        
        # Match the lengths by truncating the longer list or padding the shorter list
        if len(true_labels) < len(predicted_labels):
            predicted_labels = predicted_labels[:len(true_labels)]
        else:
            # Pad with "O"
            predicted_labels = predicted_labels + ["O"] * (len(true_labels) - len(predicted_labels))
    
    # Compute metrics
    token_metrics = compute_token_level_metrics(true_labels, predicted_labels)
    entity_metrics = compute_entity_level_metrics(true_labels, predicted_labels)
    entity_type_metrics = compute_entity_type_metrics(true_labels, predicted_labels)
    
    return {
        'token_metrics': token_metrics,
        'entity_metrics': entity_metrics,
        'entity_type_metrics': entity_type_metrics
    }


def evaluate_dataset(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate a dataset of annotated samples.
    
    Args:
        dataset: List of sample dictionaries with 'labels' and 'predicted_labels' keys
        
    Returns:
        Dictionary with evaluation metrics
    """
    all_true_labels = []
    all_pred_labels = []
    sample_metrics = []
    errors = []
    
    for i, sample in enumerate(dataset):
        if 'error' in sample or not sample.get('predicted_labels'):
            # Skip samples with errors
            errors.append(f"Sample {i}: Missing predictions or has error")
            continue
            
        try:
            true_labels = sample['labels']
            pred_labels = sample['predicted_labels']
            
            # Check for non-string labels
            for j, label in enumerate(pred_labels):
                if not isinstance(label, str):
                    print(f"Warning: Non-string label in sample {i}, position {j}: {label} (type: {type(label)})")
                    pred_labels[j] = str(label)
            
            # Verify lengths match
            if len(true_labels) != len(pred_labels):
                print(f"Warning: Length mismatch in sample {i}: true={len(true_labels)}, pred={len(pred_labels)}")
                
                # Match the lengths by truncating or padding
                if len(true_labels) < len(pred_labels):
                    pred_labels = pred_labels[:len(true_labels)]
                else:
                    pred_labels = pred_labels + ["O"] * (len(true_labels) - len(pred_labels))
                    
                # Update the sample with fixed predictions
                sample['predicted_labels'] = pred_labels
            
            all_true_labels.extend(true_labels)
            all_pred_labels.extend(pred_labels)
            
            sample_metrics.append(evaluate_sample(sample))
            
        except Exception as e:
            print(f"Error evaluating sample {i}: {str(e)}")
            errors.append(f"Sample {i}: {str(e)}")
    
    if not all_true_labels or not all_pred_labels:
        print("Warning: No valid predictions found for evaluation")
        return {
            'token_metrics': {
                'token_accuracy': 0.0,
                'token_precision': 0.0,
                'token_recall': 0.0,
                'token_f1': 0.0
            },
            'entity_metrics': {
                'entity_precision': 0.0,
                'entity_recall': 0.0,
                'entity_f1': 0.0,
                'entity_tp': 0,
                'entity_fp': 0,
                'entity_fn': 0
            },
            'errors': errors
        }
    
    # Aggregate metrics
    token_metrics = compute_token_level_metrics(all_true_labels, all_pred_labels)
    entity_metrics = compute_entity_level_metrics(all_true_labels, all_pred_labels)
    entity_type_metrics = compute_entity_type_metrics(all_true_labels, all_pred_labels)
    
    # Classification report for more detailed metrics
    true_labels_str = ensure_string_labels(all_true_labels)
    pred_labels_str = ensure_string_labels(all_pred_labels)
    report = classification_report(true_labels_str, pred_labels_str, output_dict=True)
    
    return {
        'token_metrics': token_metrics,
        'entity_metrics': entity_metrics,
        'entity_type_metrics': entity_type_metrics,
        'classification_report': report,
        'sample_metrics': sample_metrics,
        'errors': errors
    }


def print_evaluation_results(results: Dict[str, Any]) -> None:
    """
    Print evaluation results in a readable format.
    
    Args:
        results: Dictionary with evaluation metrics
    """
    if 'errors' in results and results['errors']:
        print("\n" + "=" * 80)
        print("ERRORS")
        print("=" * 80)
        for error in results['errors']:
            print(f"- {error}")
    
    token_metrics = results['token_metrics']
    entity_metrics = results['entity_metrics']
    entity_type_metrics = results['entity_type_metrics']
    
    print("=" * 80)
    print("TOKEN-LEVEL METRICS")
    print("=" * 80)
    print(f"Accuracy:  {token_metrics['token_accuracy']:.4f}")
    print(f"Precision: {token_metrics['token_precision']:.4f}")
    print(f"Recall:    {token_metrics['token_recall']:.4f}")
    print(f"F1 Score:  {token_metrics['token_f1']:.4f}")
    
    print("\n" + "=" * 80)
    print("ENTITY-LEVEL METRICS")
    print("=" * 80)
    print(f"Precision: {entity_metrics['entity_precision']:.4f}")
    print(f"Recall:    {entity_metrics['entity_recall']:.4f}")
    print(f"F1 Score:  {entity_metrics['entity_f1']:.4f}")
    print(f"True Positives:  {entity_metrics['entity_tp']}")
    print(f"False Positives: {entity_metrics['entity_fp']}")
    print(f"False Negatives: {entity_metrics['entity_fn']}")
    
    print("\n" + "=" * 80)
    print("ENTITY-TYPE METRICS")
    print("=" * 80)
    print(f"{'Entity Type':<20} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Support':<10}")
    print("-" * 80)
    for entity_type, metrics in sorted(entity_type_metrics.items()):
        print(f"{entity_type:<20} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1']:<10.4f} {metrics['support']:<10}")
    
    if 'classification_report' in results:
        print("\n" + "=" * 80)
        print("CLASSIFICATION REPORT")
        print("=" * 80)
        report = results['classification_report']
        for label, metrics in sorted(report.items()):
            if isinstance(metrics, dict) and 'precision' in metrics:
                print(f"{label:<20} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} {metrics['f1-score']:<10.4f} {metrics['support']:<10}") 