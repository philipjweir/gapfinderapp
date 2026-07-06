"""
GapFinder Web Application
-------------------------
Run with:  python app.py
Then open: http://localhost:8080
"""

import os
import json
import traceback
from pathlib import Path
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify

from pipeline import run_pipeline

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["UPLOAD_FOLDER"]      = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024   # 50 MB
app.config["SECRET_KEY"]         = "gapfinder-dev-key"

ALLOWED = {".xlsx", ".xls"}

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


def allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/analyse", methods=["POST"])
def analyse():
    # ── Validate uploads ───────────────────────────────────────────────────────
    if "responses" not in request.files or "answers" not in request.files:
        return jsonify({"error": "Both files are required."}), 400

    resp_file = request.files["responses"]
    ans_file  = request.files["answers"]

    if resp_file.filename == "" or ans_file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed(resp_file.filename) or not allowed(ans_file.filename):
        return jsonify({"error": "Only .xlsx / .xls files are accepted."}), 400

    # ── Save uploads ───────────────────────────────────────────────────────────
    upload_dir  = app.config["UPLOAD_FOLDER"]
    resp_path   = os.path.join(upload_dir, secure_filename("responses_upload.xlsx"))
    ans_path    = os.path.join(upload_dir, secure_filename("answers_upload.xlsx"))
    resp_file.save(resp_path)
    ans_file.save(ans_path)

    # ── Model path (optional) ──────────────────────────────────────────────────
    model_path = request.form.get("model_path", "").strip() or None
    if model_path and not os.path.exists(model_path):
        # Try relative to app directory
        local = os.path.join(os.path.dirname(__file__), model_path)
        model_path = local if os.path.exists(local) else None

    # ── Run pipeline ───────────────────────────────────────────────────────────
    try:
        results = run_pipeline(resp_path, ans_path, model_path)
        return jsonify(results)
    except Exception:
        tb = traceback.format_exc()
        print(tb)
        return jsonify({"error": f"Pipeline error: {tb}"}), 500


@app.route("/api/download", methods=["POST"])
def download():
    """Return CSV of failing candidates' gap profiles."""
    import io
    import csv
    from flask import Response

    data = request.get_json()
    if not data or "failing_candidates" not in data:
        return jsonify({"error": "No data supplied."}), 400

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "Student ID", "Overall Score", "Overall Result",
        "Profile Type", "N Gaps",
        "Rank", "Learning Outcome", "Mastery Probability (%)", "Severity"
    ])

    for cand in data["failing_candidates"]:
        for rank, gap in enumerate(cand["gaps"], 1):
            writer.writerow([
                cand["student_id"],
                cand["overall_score"],
                cand["overall_result"],
                cand["profile_type"],
                cand["n_gaps"],
                rank,
                gap["outcome"],
                gap["mastery_prob"],
                gap["severity"],
            ])

    buf.seek(0)
    return Response(
        buf.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=gapfinder_profiles.csv"},
    )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n  GapFinder is running at  http://localhost:8080\n")
    app.run(debug=True, port=8080)
