from flask import Flask, render_template, request, redirect, url_for
from n import HiringSystem

app = Flask(__name__)
system = HiringSystem()

@app.route('/')
def index():
    stats = system.db.get_stats()
    return render_template('index.html', stats=stats)

@app.route('/process_job', methods=['POST'])
def process_job():
    job_desc = request.form['job_description']
    jd_data = system.jd_agent.process_job_description(job_desc)
    jd_id = system.db.store_job_description(jd_data)
    return redirect(url_for('job_view', job_id=jd_id))

@app.route('/job/<int:job_id>')
def job_view(job_id):
    job = system.db.get_job_description(job_id)
    matches = system.db.get_matches_for_job(job_id)
    return render_template('job.html', job=job, matches=matches)

@app.route('/process_candidate/<int:job_id>', methods=['POST'])
def process_candidate(job_id):
    cv_text = request.form['cv_text']
    cv_data = system.cv_agent.process_cv(cv_text)
    candidate_id = system.db.store_candidate(cv_data)
    system.match_candidate_to_job(job_id, candidate_id)
    return redirect(url_for('job_view', job_id=job_id))

if __name__ == '__main__':
    app.run(debug=True, port=5001)
