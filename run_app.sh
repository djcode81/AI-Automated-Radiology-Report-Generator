#!/bin/bash
# Script to run the Streamlit app for AI Automated Report Generator

# Run Streamlit app
cd "$(dirname "$0")"
streamlit run app.py --server.port=9000 --server.maxUploadSize=1024
