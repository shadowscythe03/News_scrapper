#!/usr/bin/env python3
"""
Generate pipeline report for GitHub Actions
"""

import json
import os
from datetime import datetime

def generate_report():
    """Generate a pipeline report"""
    print("📊 Generating pipeline report...")
    
    # Read evaluation results
    report = {
        'timestamp': datetime.now().isoformat(),
        'status': 'unknown'
    }
    
    if os.path.exists('evaluation/model_evaluation.json'):
        try:
            with open('evaluation/model_evaluation.json', 'r') as f:
                eval_data = json.load(f)
            
            report.update({
                'accuracy': eval_data.get('accuracy', 'N/A'),
                'macro_f1': eval_data.get('macro_f1', 'N/A'),
                'status': 'success'
            })
            print(f"✅ Evaluation results loaded: Accuracy={report['accuracy']}, F1={report['macro_f1']}")
        except Exception as e:
            print(f"⚠️ Error reading evaluation file: {e}")
            report['status'] = 'evaluation_error'
    else:
        print("⚠️ No evaluation file found")
        report['status'] = 'no_evaluation_file'
    
    # Check if models exist
    model_files = [
        'models/sklearn_classifier.joblib',
        'models/tfidf_vectorizer.joblib',
        'models/label_encoder.joblib'
    ]
    
    models_exist = all(os.path.exists(f) for f in model_files)
    report['models_exist'] = models_exist
    
    if models_exist:
        print("✅ All ML models found")
    else:
        print("⚠️ Some ML models missing")
    
    # Save report
    with open('pipeline_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("📄 Pipeline report saved to pipeline_report.json")
    print(f"📊 Final Report: {json.dumps(report, indent=2)}")

if __name__ == "__main__":
    generate_report()