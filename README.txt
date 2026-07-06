GapFinder Web Application
=========================

SETUP
-----
1. Install dependencies (Python 3.10+):

     pip install -r requirements.txt

2. (Optional) Copy your trained model weights into this folder:

     gapfinder_qnn_best.pt

   Without the model the app uses the proportion-correct fallback.

RUN
---
     python app.py

Then open http://localhost:5000 in your browser.

USAGE
-----
1. Upload the full-results Excel file (candidate responses)
2. Upload the answer-key Excel file
3. Optionally enter the path to gapfinder_qnn_best.pt
4. Click "Analyse Responses"
5. Browse gap profiles for all failing candidates
6. Download results as CSV with the Download button

FILE STRUCTURE
--------------
gapfinder_web/
  app.py          - Flask application
  pipeline.py     - Data processing + inference logic
  requirements.txt
  README.txt
  templates/
    index.html    - Single-page educator UI
  uploads/        - Temporary file storage (auto-created)
