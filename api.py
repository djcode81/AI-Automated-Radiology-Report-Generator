from flask import Flask, request, jsonify
import os
import tempfile
import shutil
import sys
import subprocess

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pipeline import process_nifti_to_report

app = Flask(__name__)

@app.route('/generate-report', methods=['POST'])
def generate_report():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not file.filename.endswith('.nii.gz'):
        return jsonify({'error': 'File must be .nii.gz format'}), 400
    
    temp_dir = tempfile.mkdtemp()
    try:
        temp_file_path = os.path.join(temp_dir, file.filename)
        file.save(temp_file_path)
        
        cmd = ["./run_ct_pipeline.sh", temp_file_path] 
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return jsonify({'error': f'Pipeline script failed: {result.stderr}'}), 500
        
        report_path = result.stdout.strip()
        
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report = f.read()
            return jsonify({'report': report})
        else:
            return jsonify({'error': 'Report file not found'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        shutil.rmtree(temp_dir)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
