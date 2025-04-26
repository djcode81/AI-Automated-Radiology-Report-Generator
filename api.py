from flask import Flask, request, jsonify
import os
import tempfile
import shutil
import sys

# Add parent directory to path to import pipeline.py
sys.path.append('/net/dali/home/mscbio/dhp72/DS/new1')
from pipeline import process_nifti_to_report

app = Flask(__name__)

# Replace the entire generate_report function with this:
# Replace only the generate_report function in api.py with this:
# Replace the entire generate_report function with this:
@app.route('/generate-report', methods=['POST'])
def generate_report():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not file.filename.endswith('.nii.gz'):
        return jsonify({'error': 'File must be .nii.gz format'}), 400
    
    # Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    try:
        # Save the uploaded file
        temp_file_path = os.path.join(temp_dir, file.filename)
        file.save(temp_file_path)
        
        # Run our shell script
        cmd = ["/net/dali/home/mscbio/dhp72/DS/new1/run_ct_pipeline.sh", temp_file_path]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return jsonify({'error': f'Pipeline script failed: {result.stderr}'}), 500
        
        # Get the report path from the output
        report_path = result.stdout.strip()
        
        # Read the report
        if os.path.exists(report_path):
            with open(report_path, 'r') as f:
                report = f.read()
            return jsonify({'report': report})
        else:
            return jsonify({'error': 'Report file not found'}), 500
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Clean up
        shutil.rmtree(temp_dir)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
