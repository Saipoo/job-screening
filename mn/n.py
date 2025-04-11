# Multi-Agent Job Screening System
# A modular system for automated job screening with six specialized agents

import os
import re
import json
import sqlite3
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
from pathlib import Path

# Ensure NLTK data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('chunkers/maxent_ne_chunker')
    nltk.data.find('corpora/words')
except LookupError:
    print("Downloading NLTK data...")
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')

class JDSummarizerAgent:
    """Agent that parses job descriptions and extracts requirements"""
    
    def __init__(self):
        self.nltk_tokenize = word_tokenize
        self.nltk_pos_tag = pos_tag
        self.nltk_ne_chunk = ne_chunk
        
    def extract_skills(self, text):
        """Extract skills from text using NLTK"""
        tokens = self.nltk_tokenize(text)
        pos_tags = self.nltk_pos_tag(tokens)
        chunked = self.nltk_ne_chunk(pos_tags)
        
        skill_patterns = [
            "experience in", "knowledge of", "familiarity with", 
            "proficiency in", "expertise in", "skills in", "ability to",
            "years of", "background in", "degree in"
        ]
        
        skills = []
        
        # Extract named entities that might be skills
        for chunk in chunked:
            if hasattr(chunk, 'label') and chunk.label() in ["ORGANIZATION", "GPE"]:
                skills.append(' '.join(c[0] for c in chunk.leaves()).lower())
        
        # Look for noun phrases following skill patterns
        text_lower = text.lower()
        for pattern in skill_patterns:
            pattern_start = 0
            while pattern_start < len(text_lower):
                start = text_lower.find(pattern, pattern_start)
                if start == -1:
                    break
                end = start + len(pattern) + 50  # Look ahead 50 chars
                phrase = text_lower[start:end].strip()
                skills.append(phrase)
                pattern_start = start + 1
        
        # Add context-specific skills
        context_skills = self._extract_context_skills(text)
        skills.extend(context_skills)
        
        # Remove duplicates and clean up
        skills = list(set(skills))
        skills = [re.sub(r'[^\w\s]', '', skill).strip() for skill in skills]
        skills = [skill for skill in skills if len(skill) > 2]
        
        return skills
    
    def _extract_context_skills(self, text):
        """Extract skills based on common patterns in job descriptions"""
        skills = []
        
        # Look for skills in lists (often marked with bullet points)
        list_pattern = r'[•\-*]\s*(.*?)(?:\n|$)'
        for match in re.finditer(list_pattern, text):
            item = match.group(1).strip()
            if 3 < len(item) < 100:  # Reasonable skill length
                skills.append(item.lower())
        
        # Look for technical skills, languages, frameworks, etc.
        technical_patterns = [
            r'(?:python|java|javascript|typescript|c\+\+|ruby|php|sql|nosql)',
            r'(?:django|flask|react|angular|vue|node\.js|express)',
            r'(?:aws|azure|gcp|cloud)',
            r'(?:docker|kubernetes|jenkins|ci/cd)'
        ]
        
        for pattern in technical_patterns:
            for match in re.finditer(pattern, text.lower()):
                skills.append(match.group(0))
        
        return skills
    
    def extract_qualifications(self, text):
        """Extract education and experience requirements"""
        qualifications = {}
        
        # Extract education requirements
        edu_patterns = [
            r'(?:bachelor|master|phd|doctorate|degree|bs|ms|ba|ma)(?:\s+in\s+|\s+of\s+|\s+)(?:[a-z\s,]+)',
            r'(?:education|degree)(?:\s+in\s+|\:\s+|\s+requirement\:?)(?:[a-z\s,]+)'
        ]
        
        education = []
        for pattern in edu_patterns:
            for match in re.finditer(pattern, text.lower()):
                education.append(match.group(0))
        
        qualifications['education'] = list(set(education))
        
        # Extract experience requirements
        exp_patterns = [
            r'(?:\d+)(?:[\+\-])?(?:\s+years?\s+of\s+experience)',
            r'(?:experience)(?:\s+in\s+|\:\s+|\s+with\s+)(?:[a-z\s,]+)'
        ]
        
        experience = []
        for pattern in exp_patterns:
            for match in re.finditer(pattern, text.lower()):
                experience.append(match.group(0))
        
        qualifications['experience'] = list(set(experience))
        
        return qualifications
    
    def process_job_description(self, jd_text):
        """Process a job description and extract structured information"""
        # Extract job title
        title_pattern = r'^.*?(?:\n|$)'
        title_match = re.search(title_pattern, jd_text)
        title = title_match.group(0).strip() if title_match else "Unknown Position"
        
        # Process the JD text
        skills = self.extract_skills(jd_text)
        qualifications = self.extract_qualifications(jd_text)
        
        # Create structured JD object
        jd_data = {
            "title": title,
            "skills": skills,
            "qualifications": qualifications,
            "raw_text": jd_text,
            "processed_text": self._preprocess_text(jd_text)
        }
        
        return jd_data
    
    def _preprocess_text(self, text):
        """Clean and normalize text for better matching"""
        # Remove extra whitespace and convert to lowercase
        text = re.sub(r'\s+', ' ', text).lower().strip()
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\-\+\#]', '', text)
        
        return text


class CVParserAgent:
    """Agent that analyzes resumes and extracts relevant information"""
    
    def __init__(self):
        self.nltk_tokenize = word_tokenize
        self.nltk_pos_tag = pos_tag
        self.nltk_ne_chunk = ne_chunk
        
    def extract_skills(self, text):
        """Extract skills from CV text"""
        tokens = self.nltk_tokenize(text)
        pos_tags = self.nltk_pos_tag(tokens)
        chunked = self.nltk_ne_chunk(pos_tags)
        
        skills = []
        
        # Extract technical skills (often in lists or dedicated sections)
        skill_section = self._find_section(text, ["skills", "technical skills", "competencies"])
        if skill_section:
            # Extract items from lists
            list_pattern = r'[•\-*]\s*(.*?)(?:\n|$)'
            for match in re.finditer(list_pattern, skill_section):
                item = match.group(1).strip()
                if len(item) > 2:
                    skills.append(item.lower())
            
            # Extract comma-separated skills
            comma_pattern = r'(?:skills|technologies|tools)(?:\s*\:)?\s*([\w\s\,\-\+\/]+)(?:\n|$)'
            for match in re.finditer(comma_pattern, skill_section.lower()):
                items = match.group(1).split(',')
                for item in items:
                    if len(item.strip()) > 2:
                        skills.append(item.strip())
        
        # Look for skills in experience section as well
        exp_section = self._find_section(text, ["experience", "work experience", "employment"])
        if exp_section:
            # Look for technical terms, tools, languages
            tech_pattern = r'\b(?:python|java|javascript|typescript|c\+\+|ruby|php|sql|nosql|aws|azure|gcp|cloud|docker|kubernetes|jenkins|ci/cd|tensorflow|pytorch|machine learning|data science|api|rest|graphql|agile|scrum|kanban)\b'
            for match in re.finditer(tech_pattern, exp_section.lower()):
                skills.append(match.group(0))
        
        # Clean and deduplicate
        skills = list(set(skills))
        skills = [re.sub(r'[^\w\s\-\+]', '', skill).strip() for skill in skills]
        skills = [skill for skill in skills if len(skill) > 2]
        
        return skills
    
    def extract_education(self, text):
        """Extract education information from CV"""
        education = []
        
        # Find education section
        edu_section = self._find_section(text, ["education", "academic", "qualification"])
        
        if edu_section:
            # Look for degree patterns
            degree_patterns = [
                r'(?:bachelor|master|phd|doctorate|degree|bs|ms|ba|ma)\s+(?:of|in)\s+[\w\s]+',
                r'(?:university|college|institute|school)\s+of\s+[\w\s]+',
                r'[\w\s]+(?:university|college|institute|school)'
            ]
            
            for pattern in degree_patterns:
                for match in re.finditer(pattern, edu_section.lower()):
                    education.append(match.group(0))
            
            # Extract years (graduation dates)
            year_pattern = r'(?:19|20)\d{2}(?:\s*\-\s*(?:19|20)\d{2}|(?:present|current))?'
            years = []
            for match in re.finditer(year_pattern, edu_section):
                years.append(match.group(0))
            
            # Combine degrees with years if possible
            if years and education:
                edu_with_years = []
                for edu in education:
                    if not any(year in edu for year in years):
                        for year in years:
                            edu_with_years.append(f"{edu} ({year})")
                    else:
                        edu_with_years.append(edu)
                education = edu_with_years
        
        return education
    
    def extract_experience(self, text):
        """Extract work experience information"""
        experience = []
        
        # Find experience section
        exp_section = self._find_section(text, ["experience", "work experience", "employment", "professional"])
        
        if exp_section:
            # Look for job titles and companies
            job_patterns = [
                r'(?:senior|junior|lead|principal)?\s*[\w\s]+(?:engineer|developer|analyst|scientist|manager|consultant)',
                r'(?:at|with)\s+([\w\s]+)'
            ]
            
            for pattern in job_patterns:
                for match in re.finditer(pattern, exp_section.lower()):
                    experience.append(match.group(0))
            
            # Look for time periods
            time_pattern = r'(?:19|20)\d{2}\s*(?:\-|to)\s*(?:(?:19|20)\d{2}|present|current)'
            for match in re.finditer(time_pattern, exp_section.lower()):
                experience.append(match.group(0))
        
        return experience
    
    def _find_section(self, text, section_names):
        """Find a section in the text based on possible section names"""
        text_lower = text.lower()
        
        for name in section_names:
            # Try to find section header patterns
            patterns = [
                rf'\b{name}\b\s*(?::|\.|\n)(.*?)(?:\n\s*\w+\s*(?::|\.|\n)|$)',  # Section until next section
                rf'\b{name}\b(.*?)(?:\n\n|$)'  # Section until double newline
            ]
            
            for pattern in patterns:
                match = re.search(pattern, text_lower, re.DOTALL)
                if match:
                    return match.group(1).strip()
        
        return ""
    
    def process_cv(self, cv_text, candidate_name=""):
        """Process a CV and extract structured information"""
        # Try to extract name if not provided
        if not candidate_name:
            # Assume name is at the beginning
            name_match = re.search(r'^([\w\s]+)(?:\n|$)', cv_text)
            candidate_name = name_match.group(1).strip() if name_match else "Unknown Candidate"
        
        # Process CV text
        skills = self.extract_skills(cv_text)
        education = self.extract_education(cv_text)
        experience = self.extract_experience(cv_text)
        
        # Create structured CV object
        cv_data = {
            "name": candidate_name,
            "skills": skills,
            "education": education,
            "experience": experience,
            "raw_text": cv_text,
            "processed_text": self._preprocess_text(cv_text)
        }
        
        return cv_data
    
    def _preprocess_text(self, text):
        """Clean and normalize text for better matching"""
        # Remove extra whitespace and convert to lowercase
        text = re.sub(r'\s+', ' ', text).lower().strip()
        
        # Remove special characters but keep important ones
        text = re.sub(r'[^\w\s\-\+\#]', '', text)
        
        return text


class MatchingAgent:
    """Agent that computes similarity between JD and CV"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words='english')
    
    def compute_similarity(self, jd_data, cv_data):
        """Compute similarity score between JD and CV"""
        # Calculate text similarity
        text_similarity = self._compute_text_similarity(jd_data["processed_text"], cv_data["processed_text"])
        
        # Calculate skill match
        skill_match = self._compute_skill_match(jd_data["skills"], cv_data["skills"])
        
        # Calculate qualification match
        qualification_match = self._compute_qualification_match(
            jd_data["qualifications"], 
            {"education": cv_data["education"], "experience": cv_data["experience"]}
        )
        
        # Weighted overall score
        overall_score = (text_similarity * 0.3) + (skill_match * 0.5) + (qualification_match * 0.2)
        
        # Scale to 0-100
        scaled_score = min(100, max(0, overall_score * 100))
        
        # Create detailed match report
        match_details = {
            "overall_score": round(scaled_score, 2),
            "text_similarity": round(text_similarity * 100, 2),
            "skill_match": round(skill_match * 100, 2),
            "qualification_match": round(qualification_match * 100, 2),
            "matched_skills": self._get_matched_skills(jd_data["skills"], cv_data["skills"]),
            "missing_skills": self._get_missing_skills(jd_data["skills"], cv_data["skills"])
        }
        
        return match_details
    
    def _compute_text_similarity(self, jd_text, cv_text):
        """Compute cosine similarity between JD and CV text"""
        # Combine texts for fitting vectorizer
        texts = [jd_text, cv_text]
        
        # Fit and transform
        try:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
            
            # Calculate cosine similarity
            cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            return cosine_sim[0][0]
        except:
            # Fallback if vectorization fails
            return 0.5  # Neutral score
    
    def _compute_skill_match(self, jd_skills, cv_skills):
        """Compute match between JD skills and CV skills"""
        if not jd_skills:
            return 0.5  # Neutral score if no skills in JD
        
        # Convert to lowercase for case-insensitive matching
        jd_skills_lower = [skill.lower() for skill in jd_skills]
        cv_skills_lower = [skill.lower() for skill in cv_skills]
        
        # Count matches
        matches = 0
        for jd_skill in jd_skills_lower:
            # Direct match
            if jd_skill in cv_skills_lower:
                matches += 1
                continue
            
            # Partial match (skill mentioned in other skills)
            for cv_skill in cv_skills_lower:
                if (jd_skill in cv_skill) or (cv_skill in jd_skill):
                    matches += 0.5
                    break
        
        # Calculate match ratio
        if len(jd_skills) > 0:
            return min(1.0, matches / len(jd_skills))
        else:
            return 0.5  # Neutral score
    
    def _compute_qualification_match(self, jd_qualifications, cv_qualifications):
        """Compute match between JD qualifications and CV qualifications"""
        # Check education match
        education_match = self._check_education_match(
            jd_qualifications.get("education", []),
            cv_qualifications.get("education", [])
        )
        
        # Check experience match
        experience_match = self._check_experience_match(
            jd_qualifications.get("experience", []),
            cv_qualifications.get("experience", [])
        )
        
        # Equal weighting
        return (education_match + experience_match) / 2
    
    def _check_education_match(self, jd_education, cv_education):
        """Check if CV education matches JD requirements"""
        if not jd_education:
            return 1.0  # Perfect match if no requirements
        
        # Convert to lowercase
        jd_edu_lower = [edu.lower() for edu in jd_education]
        cv_edu_lower = [edu.lower() for edu in cv_education]
        
        # Look for degree level matches
        degree_levels = {
            "bachelor": 1,
            "bs": 1, 
            "ba": 1,
            "undergraduate": 1,
            "master": 2,
            "ms": 2,
            "ma": 2,
            "graduate": 2,
            "phd": 3,
            "doctorate": 3,
            "doctoral": 3
        }
        
        # Find required level in JD
        jd_level = 0
        for edu in jd_edu_lower:
            for degree, level in degree_levels.items():
                if degree in edu:
                    jd_level = max(jd_level, level)
        
        # Find highest level in CV
        cv_level = 0
        for edu in cv_edu_lower:
            for degree, level in degree_levels.items():
                if degree in edu:
                    cv_level = max(cv_level, level)
        
        # Check field match
        fields = ["computer science", "engineering", "data science", 
                 "information technology", "mathematics", "statistics",
                 "business", "management", "finance", "economics",
                 "marketing", "communication"]
        
        field_match = False
        for edu in jd_edu_lower:
            for field in fields:
                if field in edu:
                    # Check if CV has same field
                    for cv_edu in cv_edu_lower:
                        if field in cv_edu:
                            field_match = True
                            break
        
        # Calculate match score
        if jd_level == 0:
            level_score = 1.0  # No specific level required
        else:
            level_score = min(1.0, cv_level / jd_level)
        
        # Final education match score
        return (level_score * 0.7) + (0.3 if field_match else 0)
    
    def _check_experience_match(self, jd_experience, cv_experience):
        """Check if CV experience matches JD requirements"""
        if not jd_experience:
            return 1.0  # Perfect match if no requirements
        
        # Look for years of experience requirement
        years_pattern = r'(\d+)(?:\+)?\s*years?'
        required_years = 0
        
        for exp in jd_experience:
            match = re.search(years_pattern, exp.lower())
            if match:
                years = int(match.group(1))
                required_years = max(required_years, years)
        
        # Estimate candidate's years of experience
        candidate_years = 0
        for exp in cv_experience:
            # Look for date ranges
            date_pattern = r'(?:19|20)(\d{2})\s*(?:\-|to)\s*(?:((?:19|20)\d{2})|present|current)'
            match = re.search(date_pattern, exp.lower())
            if match:
                start_year = int(match.group(1))
                if match.group(2):
                    end_year = int(match.group(2))
                else:
                    # "present" or "current"
                    end_year = datetime.now().year
                duration = end_year - start_year
                candidate_years += duration
        
        # Calculate experience match score
        if required_years == 0:
            return 0.7  # Default if no specific years mentioned
        else:
            return min(1.0, candidate_years / required_years)
    
    def _get_matched_skills(self, jd_skills, cv_skills):
        """Get list of matched skills between JD and CV"""
        matched_skills = []
        
        # Convert to lowercase
        jd_skills_lower = [skill.lower() for skill in jd_skills]
        cv_skills_lower = [skill.lower() for skill in cv_skills]
        
        for jd_skill in jd_skills_lower:
            # Direct match
            if jd_skill in cv_skills_lower:
                matched_skills.append(jd_skill)
                continue
            
            # Partial match
            for cv_skill in cv_skills_lower:
                if (jd_skill in cv_skill) or (cv_skill in jd_skill):
                    matched_skills.append(f"{jd_skill} (partial match with '{cv_skill}')")
                    break
        
        return matched_skills
    
    def _get_missing_skills(self, jd_skills, cv_skills):
        """Get list of required skills missing from CV"""
        missing_skills = []
        
        # Convert to lowercase
        jd_skills_lower = [skill.lower() for skill in jd_skills]
        cv_skills_lower = [skill.lower() for skill in cv_skills]
        
        for jd_skill in jd_skills_lower:
            # Check for direct or partial match
            has_match = False
            
            # Direct match
            if jd_skill in cv_skills_lower:
                has_match = True
                continue
            
            # Partial match
            for cv_skill in cv_skills_lower:
                if (jd_skill in cv_skill) or (cv_skill in jd_skill):
                    has_match = True
                    break
            
            if not has_match:
                missing_skills.append(jd_skill)
        
        return missing_skills


class ShortlisterAgent:
    """Agent that automatically selects candidates above a threshold"""
    
    def __init__(self, threshold=70.0):
        self.threshold = threshold
    
    def set_threshold(self, new_threshold):
        """Update the threshold value"""
        if 0 <= new_threshold <= 100:
            self.threshold = new_threshold
            return True
        return False
    
    def shortlist_candidates(self, candidates_with_scores):
        """Shortlist candidates based on match scores"""
        shortlisted = []
        rejected = []
        
        for candidate in candidates_with_scores:
            if candidate["match_details"]["overall_score"] >= self.threshold:
                shortlisted.append(candidate)
            else:
                rejected.append(candidate)
        
        # Sort by score (descending)
        shortlisted.sort(key=lambda x: x["match_details"]["overall_score"], reverse=True)
        
        return {
            "shortlisted": shortlisted,
            "rejected": rejected,
            "threshold": self.threshold,
            "shortlisted_count": len(shortlisted),
            "total_count": len(candidates_with_scores)
        }
    
    def generate_shortlist_report(self, shortlist_result, job_title):
        """Generate a detailed report of the shortlisting process"""
        shortlisted = shortlist_result["shortlisted"]
        rejected = shortlist_result["rejected"]
        
        report = {
            "job_title": job_title,
            "threshold": self.threshold,
            "shortlisted_count": len(shortlisted),
            "total_candidates": len(shortlisted) + len(rejected),
            "shortlisting_rate": round(len(shortlisted) * 100 / (len(shortlisted) + len(rejected)), 2) if (len(shortlisted) + len(rejected)) > 0 else 0,
            "top_candidates": [],
            "summary": {}
        }
        
        # Add top candidates (max 5)
        for candidate in shortlisted[:5]:
            report["top_candidates"].append({
                "name": candidate["cv_data"]["name"],
                "score": candidate["match_details"]["overall_score"],
                "matched_skills": len(candidate["match_details"]["matched_skills"]),
                "missing_skills": len(candidate["match_details"]["missing_skills"])
            })
        
        # Generate summary statistics
        if shortlisted:
            scores = [c["match_details"]["overall_score"] for c in shortlisted]
            report["summary"] = {
                "average_score": round(sum(scores) / len(scores), 2),
                "highest_score": round(max(scores), 2),
                "lowest_score": round(min(scores), 2)
            }
        
        return report


class SchedulerAgent:
    """Agent that generates interview invitations and schedules"""
    
    def __init__(self, company_name="Your Company", sender_email="hr@example.com"):
        self.company_name = company_name
        self.sender_email = sender_email
    
    def generate_invitation(self, candidate, job_title, interview_date=None, interview_type="video"):
        """Generate personalized interview invitation"""
        # Get candidate name
        candidate_name = candidate["cv_data"]["name"]
        
        # Generate interview dates if not provided
        if not interview_date:
            # Generate dates for next week
            today = datetime.now()
            days_to_add = (7 - today.weekday()) % 7 + 1  # Next Monday
            interview_date = today + pd.Timedelta(days=days_to_add)
            interview_date_str = interview_date.strftime("%A, %B %d, %Y")
            interview_time_slots = [
                "10:00 AM - 11:00 AM",
                "2:00 PM - 3:00 PM",
                "4:00 PM - 5:00 PM"
            ]
        else:
            interview_date_str = interview_date
            interview_time_slots = ["As discussed"]
        
        # Create email subject
        subject = f"Interview Invitation: {job_title} Position at {self.company_name}"
        
        # Create email body
        body = f"""Dear {candidate_name},

We are pleased to inform you that after reviewing your application for the {job_title} position at {self.company_name}, we would like to invite you for an interview.

Your profile has matched with the job requirements, and we believe your skills and experience would be valuable to our team.

Interview Details:
- Position: {job_title}
- Date: {interview_date_str}
- Format: {interview_type.capitalize()} Interview

Available Time Slots:
"""
        
        # Add time slots
        for i, slot in enumerate(interview_time_slots, 1):
            body += f"{i}. {slot}\n"
        
        body += f"""
Please reply to this email with your preferred time slot, and we will confirm your interview schedule.

If none of these times work for you, please suggest alternative times, and we will do our best to accommodate your schedule.

Before the interview, please take some time to:
1. Research our company and the position
2. Prepare examples of your past work relevant to this role
3. Have questions ready about the role and our company

We look forward to speaking with you!

Best regards,
Recruitment Team
{self.company_name}
{self.sender_email}
"""
        
        # Create email structure
        email = {
            "to": f"{candidate_name} <candidate@example.com>",
            "from": f"Recruitment Team <{self.sender_email}>",
            "subject": subject,
            "body": body
        }
        
        return email
    
    def send_email(self, email_data, smtp_config=None):
        """Send email using SMTP (demonstration only - not actually sending)"""
        # This function is for demonstration purposes
        # In a real application, you would connect to an SMTP server
        
        # Create mime message
        message = MIMEMultipart()
        message["From"] = email_data["from"]
        message["To"] = email_data["to"]
        message["Subject"] = email_data["subject"]
        
        # Attach body
        message.attach(MIMEText(email_data["body"], "plain"))
        
        # Log instead of sending
        print(f"[MOCK EMAIL] Would send email to: {email_data['to']}")
        print(f"[MOCK EMAIL] Subject: {email_data['subject']}")
        print(f"[MOCK EMAIL] Body length: {len(email_data['body'])} characters")
        
        # Return success
        return {
            "success": True,
            "message": "Email prepared successfully (not actually sent)"
        }
    
    def generate_schedule(self, shortlisted_candidates, job_title, start_date=None, interview_duration=60):
        """Generate an interview schedule for shortlisted candidates"""
        # Default to starting next Monday if no date provided
        if not start_date:
            today = datetime.now()
            days_to_add = (7 - today.weekday()) % 7 + 1  # Next Monday
            start_date = today + pd.Timedelta(days=days_to_add)
        elif isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        
        # Create schedule
        schedule = []
        
        # Set working hours (9 AM to 5 PM)
        work_start_hour = 9
        work_end_hour = 17
        
        current_date = start_date
        current_hour = work_start_hour
        
        for candidate in shortlisted_candidates:
            # Skip weekends
            while current_date.weekday() >= 5:  # 5=Saturday, 6=Sunday
                current_date += pd.Timedelta(days=1)
                current_hour = work_start_hour
            
            # Format interview time
            interview_start = pd.Timestamp(
                year=current_date.year,
                month=current_date.month,
                day=current_date.day,
                hour=int(current_hour),
                minute=int((current_hour % 1) * 60)
            )
            
            interview_end = interview_start + pd.Timedelta(minutes=interview_duration)
            
            # Add to schedule
            schedule.append({
                "candidate_name": candidate["cv_data"]["name"],
                "job_title": job_title,
                "interview_start": interview_start.strftime("%Y-%m-%d %H:%M"),
                "interview_end": interview_end.strftime("%Y-%m-%d %H:%M"),
                "score": candidate["match_details"]["overall_score"],
                "interviewer": "HR Team"
            })
            
            # Update time for next interview (add 30 min buffer)
            current_hour += (interview_duration + 30) / 60
            
            # Move to next day if work day is over
            if current_hour >= work_end_hour:
                current_date += pd.Timedelta(days=1)
                current_hour = work_start_hour
        
        return {
            "job_title": job_title,
            "total_interviews": len(schedule),
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": schedule[-1]["interview_end"].split()[0] if schedule else start_date.strftime("%Y-%m-%d"),
            "schedule": schedule
        }


class SQLiteMemoryAgent:
    """Agent that stores and retrieves data using SQLite"""
    
    def __init__(self, db_path="job_screening_system.db"):
        self.db_path = db_path
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize the database with necessary tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create job descriptions table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS job_descriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            raw_text TEXT NOT NULL,
            processed_text TEXT NOT NULL,
            skills TEXT NOT NULL,
            qualifications TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create candidates table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            raw_text TEXT NOT NULL,
            processed_text TEXT NOT NULL,
            skills TEXT NOT NULL,
            education TEXT NOT NULL,
            experience TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create matches table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS matches (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER NOT NULL,
            candidate_id INTEGER NOT NULL,
            overall_score REAL NOT NULL,
            text_similarity REAL NOT NULL,
            skill_match REAL NOT NULL,
            qualification_match REAL NOT NULL,
            matched_skills TEXT NOT NULL,
            missing_skills TEXT NOT NULL,
            shortlisted BOOLEAN DEFAULT FALSE,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (job_id) REFERENCES job_descriptions (id),
            FOREIGN KEY (candidate_id) REFERENCES candidates (id)
        )
        ''')
        
        # Create interviews table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS interviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            match_id INTEGER NOT NULL,
            interview_start TIMESTAMP NOT NULL,
            interview_end TIMESTAMP NOT NULL,
            interview_type TEXT DEFAULT 'video',
            status TEXT DEFAULT 'scheduled',
            notes TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (match_id) REFERENCES matches (id)
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def store_job_description(self, jd_data):
        """Store job description in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO job_descriptions (title, raw_text, processed_text, skills, qualifications)
        VALUES (?, ?, ?, ?, ?)
        ''', (
            jd_data["title"],
            jd_data["raw_text"],
            jd_data["processed_text"],
            json.dumps(jd_data["skills"]),
            json.dumps(jd_data["qualifications"])
        ))
        
        job_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        return job_id
    
    def store_candidate(self, cv_data):
        """Store candidate CV in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO candidates (name, raw_text, processed_text, skills, education, experience)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            cv_data["name"],
            cv_data["raw_text"],
            cv_data["processed_text"],
            json.dumps(cv_data["skills"]),
            json.dumps(cv_data["education"]),
            json.dumps(cv_data["experience"])
        ))
        
        candidate_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        return candidate_id
    
    def store_match(self, job_id, candidate_id, match_details, shortlisted=False):
        """Store match details in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO matches (
            job_id, candidate_id, overall_score, text_similarity, 
            skill_match, qualification_match, matched_skills, 
            missing_skills, shortlisted
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            job_id,
            candidate_id,
            match_details["overall_score"],
            match_details["text_similarity"],
            match_details["skill_match"],
            match_details["qualification_match"],
            json.dumps(match_details["matched_skills"]),
            json.dumps(match_details["missing_skills"]),
            shortlisted
        ))
        
        match_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        return match_id
    
    def store_interview(self, match_id, interview_start, interview_end, interview_type="video"):
        """Store interview details in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO interviews (match_id, interview_start, interview_end, interview_type)
        VALUES (?, ?, ?, ?)
        ''', (
            match_id,
            interview_start,
            interview_end,
            interview_type
        ))
        
        interview_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        return interview_id
    
    def get_job_description(self, job_id):
        """Retrieve job description from database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM job_descriptions WHERE id = ?', (job_id,))
        row = cursor.fetchone()
        
        if row:
            jd_data = dict(row)
            # Parse JSON fields
            jd_data["skills"] = json.loads(jd_data["skills"])
            jd_data["qualifications"] = json.loads(jd_data["qualifications"])
            conn.close()
            return jd_data
        
        conn.close()
        return None
    
    def get_candidate(self, candidate_id):
        """Retrieve candidate data from database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM candidates WHERE id = ?', (candidate_id,))
        row = cursor.fetchone()
        
        if row:
            cv_data = dict(row)
            # Parse JSON fields
            cv_data["skills"] = json.loads(cv_data["skills"])
            cv_data["education"] = json.loads(cv_data["education"])
            cv_data["experience"] = json.loads(cv_data["experience"])
            conn.close()
            return cv_data
        
        conn.close()
        return None
    
    def get_matches_for_job(self, job_id, shortlisted_only=False):
        """Get all candidate matches for a job"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        if shortlisted_only:
            query = '''
            SELECT m.*, c.name as candidate_name, j.title as job_title
            FROM matches m
            JOIN candidates c ON m.candidate_id = c.id
            JOIN job_descriptions j ON m.job_id = j.id
            WHERE m.job_id = ? AND m.shortlisted = 1
            ORDER BY m.overall_score DESC
            '''
        else:
            query = '''
            SELECT m.*, c.name as candidate_name, j.title as job_title
            FROM matches m
            JOIN candidates c ON m.candidate_id = c.id
            JOIN job_descriptions j ON m.job_id = j.id
            WHERE m.job_id = ?
            ORDER BY m.overall_score DESC
            '''
        
        cursor.execute(query, (job_id,))
        rows = cursor.fetchall()
        
        matches = []
        for row in rows:
            match_data = dict(row)
            # Parse JSON fields
            match_data["matched_skills"] = json.loads(match_data["matched_skills"])
            match_data["missing_skills"] = json.loads(match_data["missing_skills"])
            matches.append(match_data)
        
        conn.close()
        return matches
    
    def get_interviews(self, job_id=None, candidate_id=None, status=None):
        """Get interviews with optional filters"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = '''
        SELECT i.*, m.overall_score, c.name as candidate_name, j.title as job_title
        FROM interviews i
        JOIN matches m ON i.match_id = m.id
        JOIN candidates c ON m.candidate_id = c.id
        JOIN job_descriptions j ON m.job_id = j.id
        WHERE 1=1
        '''
        
        params = []
        
        if job_id:
            query += ' AND m.job_id = ?'
            params.append(job_id)
        
        if candidate_id:
            query += ' AND m.candidate_id = ?'
            params.append(candidate_id)
        
        if status:
            query += ' AND i.status = ?'
            params.append(status)
        
        query += ' ORDER BY i.interview_start ASC'
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        interviews = [dict(row) for row in rows]
        
        conn.close()
        return interviews
    
    def update_interview_status(self, interview_id, status, notes=None):
        """Update interview status and notes"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if notes:
            cursor.execute('''
            UPDATE interviews SET status = ?, notes = ? WHERE id = ?
            ''', (status, notes, interview_id))
        else:
            cursor.execute('''
            UPDATE interviews SET status = ? WHERE id = ?
            ''', (status, interview_id))
        
        conn.commit()
        conn.close()
        
        return True
    
    def get_stats(self):
        """Get system statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get counts
        cursor.execute('SELECT COUNT(*) FROM job_descriptions')
        job_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM candidates')
        candidate_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM matches')
        match_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM matches WHERE shortlisted = 1')
        shortlisted_count = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM interviews')
        interview_count = cursor.fetchone()[0]
        
        # Get average scores
        cursor.execute('SELECT AVG(overall_score) FROM matches')
        avg_score = cursor.fetchone()[0] or 0
        
        # Get top skills
        cursor.execute('''
        SELECT skills FROM job_descriptions
        ''')
        all_job_skills = cursor.fetchall()
        
        # Close connection
        conn.close()
        
        # Process skills
        skill_counts = {}
        for skills_json in all_job_skills:
            skills = json.loads(skills_json[0])
            for skill in skills:
                skill_counts[skill.lower()] = skill_counts.get(skill.lower(), 0) + 1
        
        top_skills = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        stats = {
            "job_count": job_count,
            "candidate_count": candidate_count,
            "match_count": match_count,
            "shortlisted_count": shortlisted_count,
            "interview_count": interview_count,
            "average_score": round(avg_score, 2) if avg_score else 0,
            "shortlisting_rate": round(shortlisted_count * 100 / match_count, 2) if match_count > 0 else 0,
            "top_skills": [{"skill": skill, "count": count} for skill, count in top_skills]
        }
        
        return stats


class JobScreeningSystem:
    """Main system that coordinates all agents"""
    
    def __init__(self, db_path="job_screening_system.db", company_name="Your Company"):
        self.jd_agent = JDSummarizerAgent()
        self.cv_agent = CVParserAgent()
        self.matching_agent = MatchingAgent()
        self.shortlister_agent = ShortlisterAgent(threshold=70.0)
        self.scheduler_agent = SchedulerAgent(company_name=company_name)
        self.db_agent = SQLiteMemoryAgent(db_path=db_path)
    
    def process_job_description(self, jd_text):
        """Process a job description and store it"""
        # Parse JD
        jd_data = self.jd_agent.process_job_description(jd_text)
        
        # Store in database
        job_id = self.db_agent.store_job_description(jd_data)
        
        # Return processed JD with ID
        jd_data["id"] = job_id
        
        return jd_data
    
    def process_candidate_cv(self, cv_text, candidate_name=None):
        """Process a candidate CV and store it"""
        # Parse CV
        cv_data = self.cv_agent.process_cv(cv_text, candidate_name)
        
        # Store in database
        candidate_id = self.db_agent.store_candidate(cv_data)
        
        # Return processed CV with ID
        cv_data["id"] = candidate_id
        
        return cv_data
    
    def match_candidate_to_job(self, job_id, candidate_id, auto_shortlist=True):
        """Match a candidate to a job and optionally shortlist"""
        # Get job and candidate data
        jd_data = self.db_agent.get_job_description(job_id)
        cv_data = self.db_agent.get_candidate(candidate_id)
        
        if not jd_data or not cv_data:
            return {"error": "Job or candidate not found"}
        
        # Compute match
        match_details = self.matching_agent.compute_similarity(jd_data, cv_data)
        
        # Determine if shortlisted
        shortlisted = auto_shortlist and match_details["overall_score"] >= self.shortlister_agent.threshold
        
        # Store match in database
        match_id = self.db_agent.store_match(job_id, candidate_id, match_details, shortlisted)
        
        # Return match details with IDs
        return {
            "match_id": match_id,
            "job_id": job_id,
            "candidate_id": candidate_id,
            "shortlisted": shortlisted,
            "match_details": match_details
        }
    
    def batch_process_candidates(self, job_id, cv_list):
        """Process multiple candidates for a job"""
        results = []
        
        # Get job data
        jd_data = self.db_agent.get_job_description(job_id)
        
        if not jd_data:
            return {"error": "Job not found"}
        
        # Process each CV
        for cv_item in cv_list:
            cv_text = cv_item.get("text", "")
            candidate_name = cv_item.get("name", None)
            
            # Parse CV
            cv_data = self.cv_agent.process_cv(cv_text, candidate_name)
            
            # Store in database
            candidate_id = self.db_agent.store_candidate(cv_data)
            cv_data["id"] = candidate_id
            
            # Match with job
            match_details = self.matching_agent.compute_similarity(jd_data, cv_data)
            
            # Determine if shortlisted
            shortlisted = match_details["overall_score"] >= self.shortlister_agent.threshold
            
            # Store match in database
            match_id = self.db_agent.store_match(job_id, candidate_id, match_details, shortlisted)
            
            # Add to results
            results.append({
                "match_id": match_id,
                "job_id": job_id,
                "candidate_id": candidate_id,
                "candidate_name": cv_data["name"],
                "shortlisted": shortlisted,
                "overall_score": match_details["overall_score"],
                "cv_data": cv_data,
                "match_details": match_details
            })
        
        # Sort by score
        results.sort(key=lambda x: x["overall_score"], reverse=True)
        
        return results
    
    def generate_shortlist(self, job_id, custom_threshold=None):
        """Generate a shortlist for a job"""
        # Update threshold if provided
        if custom_threshold is not None:
            self.shortlister_agent.set_threshold(custom_threshold)
        
        # Get job data
        jd_data = self.db_agent.get_job_description(job_id)
        
        if not jd_data:
            return {"error": "Job not found"}
        
        # Get all matches for this job
        all_matches = self.db_agent.get_matches_for_job(job_id)
        
        # Convert to the format expected by shortlister
        candidates_with_scores = []
        for match in all_matches:
            cv_data = self.db_agent.get_candidate(match["candidate_id"])
            
            candidates_with_scores.append({
                "match_id": match["id"],
                "job_id": job_id,
                "candidate_id": match["candidate_id"],
                "cv_data": cv_data,
                "match_details": {
                    "overall_score": match["overall_score"],
                    "text_similarity": match["text_similarity"],
                    "skill_match": match["skill_match"],
                    "qualification_match": match["qualification_match"],
                    "matched_skills": match["matched_skills"],
                    "missing_skills": match["missing_skills"]
                }
            })
        
        # Generate shortlist
        shortlist_result = self.shortlister_agent.shortlist_candidates(candidates_with_scores)
        
        # Generate report
        shortlist_report = self.shortlister_agent.generate_shortlist_report(
            shortlist_result, jd_data["title"]
        )
        
        # Update database to mark shortlisted candidates
        for candidate in shortlist_result["shortlisted"]:
            match_id = candidate["match_id"]
            # Update match in database to mark as shortlisted
            conn = sqlite3.connect(self.db_agent.db_path)
            cursor = conn.cursor()
            cursor.execute('UPDATE matches SET shortlisted = 1 WHERE id = ?', (match_id,))
            conn.commit()
            conn.close()
        
        return {
            "shortlist_result": shortlist_result,
            "shortlist_report": shortlist_report
        }
    
    def schedule_interviews(self, job_id, interview_date=None, interview_type="video"):
        """Schedule interviews for shortlisted candidates"""
        # Get job data
        jd_data = self.db_agent.get_job_description(job_id)
        
        if not jd_data:
            return {"error": "Job not found"}
        
        # Get shortlisted candidates
        shortlisted = self.db_agent.get_matches_for_job(job_id, shortlisted_only=True)
        
        if not shortlisted:
            return {"error": "No shortlisted candidates found"}
        
        # Convert to format expected by scheduler
        shortlisted_candidates = []
        for match in shortlisted:
            cv_data = self.db_agent.get_candidate(match["candidate_id"])
            
            shortlisted_candidates.append({
                "match_id": match["id"],
                "job_id": job_id,
                "candidate_id": match["candidate_id"],
                "cv_data": cv_data,
                "match_details": {
                    "overall_score": match["overall_score"]
                }
            })
        
        # Generate interview schedule
        schedule = self.scheduler_agent.generate_schedule(
            shortlisted_candidates, jd_data["title"], interview_date
        )
        
        # Store interviews in database
        for interview in schedule["schedule"]:
            match_id = next(c["match_id"] for c in shortlisted_candidates 
                           if c["cv_data"]["name"] == interview["candidate_name"])
            
            self.db_agent.store_interview(
                match_id,
                interview["interview_start"],
                interview["interview_end"],
                interview_type
            )
        
        return schedule
    
    def send_interview_invitations(self, job_id, interview_date=None):
        """Generate and send interview invitations"""
        # Get job data
        jd_data = self.db_agent.get_job_description(job_id)
        
        if not jd_data:
            return {"error": "Job not found"}
        
        # Get shortlisted candidates
        shortlisted = self.db_agent.get_matches_for_job(job_id, shortlisted_only=True)
        
        if not shortlisted:
            return {"error": "No shortlisted candidates found"}
        
        # Send invitation to each candidate
        results = []
        for match in shortlisted:
            cv_data = self.db_agent.get_candidate(match["candidate_id"])
            
            # Generate invitation
            invitation = self.scheduler_agent.generate_invitation(
                {"cv_data": cv_data},
                jd_data["title"],
                interview_date,
                "video"
            )
            
            # Mock sending email
            send_result = self.scheduler_agent.send_email(invitation)
            
            results.append({
                "candidate_name": cv_data["name"],
                "email": invitation,
                "result": send_result
            })
        
        return {
            "job_title": jd_data["title"],
            "invitations_sent": len(results),
            "results": results
        }
    
    def get_system_stats(self):
        """Get overall system statistics"""
        return self.db_agent.get_stats()
    
    def export_shortlist(self, job_id, format="json"):
        """Export shortlist data in various formats"""
        shortlisted = self.db_agent.get_matches_for_job(job_id, shortlisted_only=True)
        
        if not shortlisted:
            return {"error": "No shortlisted candidates found"}
        
        # Get job data
        jd_data = self.db_agent.get_job_description(job_id)
        
        # Build export data
        export_data = {
            "job_title": jd_data["title"],
            "export_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "shortlisted_count": len(shortlisted),
            "candidates": []
        }
        
        for match in shortlisted:
            cv_data = self.db_agent.get_candidate(match["candidate_id"])
            
            candidate_data = {
                "name": cv_data["name"],
                "score": match["overall_score"],
                "skills": cv_data["skills"],
                "education": cv_data["education"],
                "experience": cv_data["experience"],
                "matched_skills": match["matched_skills"],
                "missing_skills": match["missing_skills"]
            }
            
            export_data["candidates"].append(candidate_data)
        
        # Format output
        if format == "json":
            return json.dumps(export_data, indent=2)
        elif format == "csv":
            # Build CSV string
            csv_data = "Candidate Name,Score,Matched Skills,Missing Skills\n"
            for candidate in export_data["candidates"]:
                matched = "|".join(candidate["matched_skills"]).replace(",", ";")
                missing = "|".join(candidate["missing_skills"]).replace(",", ";")
                csv_data += f'{candidate["name"]},{candidate["score"]},"{matched}","{missing}"\n'
            return csv_data
        else:
            return export_data


class HiringSystem:
    """Main hiring system class that handles all hiring processes"""
    def __init__(self):
        self.db = SQLiteMemoryAgent()
        self.jd_agent = JDSummarizerAgent()
        self.cv_agent = CVParserAgent()
        self.matching_agent = MatchingAgent()
        self.shortlister_agent = ShortlisterAgent()
        self.scheduler_agent = SchedulerAgent()

    def match_candidate_to_job(self, job_id, candidate_id):
        """Match a candidate to a job using the matching agent"""
        job = self.db.get_job_description(job_id)
        candidate = self.db.get_candidate(candidate_id)
        if not job or not candidate:
            return None
        match_result = self.matching_agent.compute_similarity(job, candidate)
        self.db.store_match(job_id, candidate_id, match_result)
        return match_result

# Example usage
if __name__ == "__main__":
    # Initialize system
    system = JobScreeningSystem(company_name="TechCorp Inc.")
    
    # Example job description
    jd_text = """
    Senior Python Developer
    
    We are looking for an experienced Python developer to join our team.
    
    Requirements:
    • 5+ years of experience in Python development
    • Strong knowledge of web frameworks (Django, Flask)
    • Experience with database systems (SQL, NoSQL)
    • Understanding of cloud services (AWS, Azure)
    • Experience with CI/CD pipelines
    
    Education:
    • Bachelor's degree in Computer Science or related field
    
    About us:
    We are a growing tech company focused on building scalable solutions.
    """
    
    # Example candidate CV
    cv_text = """
    John Smith
    
    Professional Summary:
    Experienced software developer with 7 years of Python development.
    
    Skills:
    • Python, Django, Flask
    • SQL (PostgreSQL, MySQL)
    • AWS (EC2, S3, Lambda)
    • Docker, Jenkins
    
    Experience:
    Python Developer | TechSolutions Inc.
    2018 - Present
    - Developed RESTful APIs using Django Rest Framework
    - Implemented CI/CD pipelines with Jenkins
    
    Junior Developer | CodeCorp
    2016 - 2018
    - Worked on Flask applications
    - Database management with PostgreSQL
    
    Education:
    Bachelor of Science in Computer Science
    University of Technology (2016)
    """
    
    # Process job description
    print("Processing job description...")
    jd_data = system.process_job_description(jd_text)
    print(f"Job ID: {jd_data['id']}")
    print(f"Extracted skills: {jd_data['skills']}")
    
    # Process candidate CV
    print("\nProcessing candidate CV...")
    cv_data = system.process_candidate_cv(cv_text)
    print(f"Candidate ID: {cv_data['id']}")
    print(f"Extracted skills: {cv_data['skills']}")
    
    # Match candidate to job
    print("\nMatching candidate to job...")
    match_result = system.match_candidate_to_job(jd_data['id'], cv_data['id'])
    print(f"Match score: {match_result['match_details']['overall_score']}")
    print(f"Matched skills: {match_result['match_details']['matched_skills']}")
    print(f"Missing skills: {match_result['match_details']['missing_skills']}")
    
    # Generate shortlist
    print("\nGenerating shortlist...")
    shortlist = system.generate_shortlist(jd_data['id'])
    print(f"Shortlisted: {shortlist['shortlist_report']['shortlisted_count']} of {shortlist['shortlist_report']['total_candidates']}")
    
    # Schedule interviews
    print("\nScheduling interviews...")
    schedule_result = system.schedule_interviews(jd_data['id'])
    interview_count = len(schedule_result.get('schedule', []))
    print(f"Scheduled {interview_count} interviews")
    
    # Send invitations
    print("\nSending interview invitations...")
    invitations = system.send_interview_invitations(jd_data['id'])
    sent_count = invitations.get('invitations_sent', 0)
    print(f"Sent {sent_count} invitations")
    
    # Get stats
    print("\nSystem statistics:")
    stats = system.get_system_stats()
    print(f"Jobs: {stats['job_count']}")
    print(f"Candidates: {stats['candidate_count']}")
    print(f"Matches: {stats['match_count']}")
    print(f"Shortlisted: {stats['shortlisted_count']}")
    print(f"Average score: {stats['average_score']}")