# Hiring System

A multi-agent job screening system with automated candidate matching and interview scheduling.

## Features

- Job description processing via JDSummarizerAgent
- CV parsing via CVParserAgent
- Candidate-job matching via MatchingAgent
- Shortlisting via ShortlisterAgent
- Interview scheduling via SchedulerAgent
- Database persistence via SQLiteMemoryAgent
- Web interface for user interaction

## Requirements

- Python 3.7+
- Flask
- NLTK
- scikit-learn
- pandas
- numpy

## Setup

1. Clone the repository
2. Install requirements: `pip install -r requirements.txt`
3. Run: `python app.py`
4. Access at: http://localhost:5001
