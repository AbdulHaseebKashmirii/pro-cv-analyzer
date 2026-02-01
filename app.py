"""
CV Analyzer Pro - Recruiter's Smart Screening Assistant
========================================================
A business-driven CV analysis tool for recruiters and hiring managers.

Features:
1. JD ↔ CV Fit Score with evidence-based matching
2. Structured Candidate Snapshot with JSON/CSV export
3. Interview Kit with role-specific questions

Built with Streamlit + Google Gemini API
"""

import streamlit as st
import PyPDF2
import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

# =============================================================================
# CONFIGURATION
# =============================================================================

# Load environment variables from .env file
load_dotenv()

# Page configuration - must be first Streamlit command
st.set_page_config(
    page_title="CV Analyzer Pro",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Google Gemini client
# API key loaded from environment variable for security
google_api_key = os.environ.get("GOOGLE_API_KEY")

if not google_api_key:
    st.error("⚠️ GOOGLE_API_KEY not found. Please set it in your .env file.")
    st.stop()

client = genai.Client(api_key=google_api_key)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def extract_text_from_pdf(uploaded_file):
    """
    Extract text content from an uploaded PDF file.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        str: Extracted text from all pages
    """
    try:
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""


def call_gemini_api(prompt):
    """
    Send a prompt to Google Gemini API and return the response.
    
    Args:
        prompt: The full prompt string to send
        
    Returns:
        str: Model response text or error message
    """
    try:
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            ),
        ]
        
        response = client.models.generate_content(
            model="models/gemini-flash-lite-latest",
            contents=contents,
        )
        return response.text
    except Exception as e:
        return f"API Error: {e}"


def generate_fit_score(cv_text, jd_text):
    """
    Generate a fit score (0-100) comparing CV against Job Description.
    Returns structured analysis with evidence in markdown format.
    """
    prompt = f"""You are an expert recruiter assistant. Analyze how well this CV matches the job description.

JOB DESCRIPTION:
{jd_text}

CANDIDATE CV:
{cv_text}

Provide your analysis in this EXACT markdown format:

## 🎯 FIT SCORE: [number 0-100]/100

### ✅ Matched Requirements
- **[requirement 1]:** [brief evidence from CV]
- **[requirement 2]:** [brief evidence from CV]
- **[requirement 3]:** [brief evidence from CV]
- **[requirement 4]:** [brief evidence from CV]
- **[requirement 5]:** [brief evidence from CV]

### ⚠️ Missing/Weak Areas
- **[gap 1]:** [what's missing or weak]
- **[gap 2]:** [what's missing or weak]

### 📋 Verdict
**[Strong Fit / Moderate Fit / Weak Fit]:** [One sentence recommendation]

Use proper markdown with headers (##, ###), bold (**text**), and bullet points (-). Keep it well-structured and readable."""

    return call_gemini_api(prompt)


def generate_candidate_snapshot(cv_text):
    """
    Extract structured candidate information from CV.
    Returns a clean, parseable format.
    """
    prompt = f"""You are a CV parsing expert. Extract key information from this CV into a structured format.

CV TEXT:
{cv_text}

Return ONLY valid JSON (no markdown, no code blocks, no extra text). Use this exact structure:
{{
    "name": "Full Name",
    "email": "email@example.com",
    "phone": "phone number or null",
    "linkedin": "LinkedIn URL or null",
    "summary": "2-3 sentence professional summary",
    "experience_years": "estimated total years",
    "current_role": "most recent job title",
    "current_company": "most recent company",
    "skills": ["skill1", "skill2", "skill3"],
    "tools_technologies": ["tool1", "tool2"],
    "education": [
        {{"degree": "Degree Name", "institution": "University", "year": "Year"}}
    ],
    "key_achievements": ["achievement 1", "achievement 2"],
    "languages": ["English", "Other"]
}}

If any field is not found in the CV, use null for single values or empty array [] for lists."""

    return call_gemini_api(prompt)


def generate_interview_kit(cv_text, jd_text):
    """
    Generate role-specific interview questions based on CV and JD.
    Includes evaluation rubrics and risk areas in markdown format.
    """
    prompt = f"""You are a senior hiring manager. Create an interview kit for this candidate.

JOB DESCRIPTION:
{jd_text}

CANDIDATE CV:
{cv_text}

Generate an interview kit in this EXACT markdown format:

## 🎤 Role-Specific Questions

### Question 1: Experience
**Q:** [Question about their relevant experience]

✅ **Strong Answer Should Include:** [What a good answer includes]

---

### Question 2: Technical Skills
**Q:** [Technical/skill-based question from JD requirements]

✅ **Strong Answer Should Include:** [What a good answer includes]

---

### Question 3: Project Deep-Dive
**Q:** [Question about a specific project from their CV]

✅ **Strong Answer Should Include:** [What a good answer includes]

---

### Question 4: Behavioral
**Q:** [Behavioral question relevant to the role]

✅ **Strong Answer Should Include:** [What a good answer includes]

---

### Question 5: Problem-Solving
**Q:** [Problem-solving question for the role]

✅ **Strong Answer Should Include:** [What a good answer includes]

---

### Question 6: Culture Fit
**Q:** [Culture/teamwork question]

✅ **Strong Answer Should Include:** [What a good answer includes]

---

## 🔍 Clarification Questions (Address Gaps/Concerns)

### Clarification 1
**Q:** [Question about a gap, unclear period, or missing requirement]

🎯 **Why Ask This:** [What you're trying to clarify]

---

### Clarification 2
**Q:** [Question about a potential risk area]

🎯 **Why Ask This:** [What you're trying to clarify]

Use proper markdown formatting with headers, bold text, and horizontal rules (---) for clear separation."""

    return call_gemini_api(prompt)


def parse_json_safely(json_string):
    """
    Safely parse JSON from API response, handling common issues.
    """
    # Clean up common issues
    cleaned = json_string.strip()
    
    # Remove markdown code blocks if present
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    
    cleaned = cleaned.strip()
    
    try:
        return json.loads(cleaned), None
    except json.JSONDecodeError as e:
        return None, str(e)


# =============================================================================
# UI COMPONENTS
# =============================================================================

# Header
st.title("📋 CV Analyzer Pro")
st.markdown("**Recruiter's Smart Screening Assistant** — Less but better.")

# Privacy notice
st.caption("🔒 Your data is processed in-memory only. We don't store CVs or job descriptions.")

st.divider()

# =============================================================================
# SIDEBAR - INPUTS
# =============================================================================

with st.sidebar:
    st.header("📄 Step 1: Upload CV")
    uploaded_file = st.file_uploader(
        "Upload candidate CV (PDF)",
        type=["pdf"],
        help="Upload a PDF resume/CV to analyze"
    )
    
    cv_text = ""
    if uploaded_file:
        with st.spinner("Extracting text..."):
            cv_text = extract_text_from_pdf(uploaded_file)
        
        if cv_text:
            st.success(f"✅ CV loaded ({len(cv_text.split())} words)")
        else:
            st.error("❌ Could not extract text from PDF")
    
    st.divider()
    
    st.header("📝 Step 2: Paste Job Description")
    jd_text = st.text_area(
        "Job Description",
        height=200,
        placeholder="Paste the job description here...\n\nExample:\nWe are looking for a Software Engineer with 3+ years of Python experience...",
        help="Paste the full job description for accurate matching"
    )
    
    st.divider()
    
    # Analysis button
    st.header("🚀 Step 3: Analyze")
    analyze_button = st.button(
        "⚡ Run Full Analysis",
        type="primary",
        use_container_width=True,
        disabled=not (cv_text and jd_text.strip())
    )
    
    if not cv_text:
        st.caption("⬆️ Upload a CV to continue")
    elif not jd_text.strip():
        st.caption("⬆️ Paste a job description")

# =============================================================================
# MAIN CONTENT - ANALYSIS RESULTS
# =============================================================================

# Initialize session state for results
if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False
    st.session_state.fit_score_result = ""
    st.session_state.snapshot_result = ""
    st.session_state.interview_kit_result = ""
    st.session_state.snapshot_json = None

# Run analysis when button clicked
if analyze_button and cv_text and jd_text.strip():
    
    # Progress tracking
    progress_bar = st.progress(0, text="Starting analysis...")
    
    # Step 1: Fit Score
    progress_bar.progress(10, text="📊 Calculating fit score...")
    st.session_state.fit_score_result = generate_fit_score(cv_text, jd_text)
    
    # Step 2: Candidate Snapshot
    progress_bar.progress(40, text="👤 Extracting candidate profile...")
    st.session_state.snapshot_result = generate_candidate_snapshot(cv_text)
    
    # Parse JSON
    parsed_json, error = parse_json_safely(st.session_state.snapshot_result)
    st.session_state.snapshot_json = parsed_json
    
    # Step 3: Interview Kit
    progress_bar.progress(70, text="📝 Generating interview questions...")
    st.session_state.interview_kit_result = generate_interview_kit(cv_text, jd_text)
    
    # Complete
    progress_bar.progress(100, text="✅ Analysis complete!")
    st.session_state.analysis_complete = True
    
    # Clear progress bar after a moment
    progress_bar.empty()

# Display results if analysis is complete
if st.session_state.analysis_complete:
    
    # Create three tabs for organized display
    tab1, tab2, tab3 = st.tabs([
        "📊 Fit Score & Evidence",
        "👤 Candidate Snapshot",
        "📝 Interview Kit"
    ])
    
    # -------------------------------------------------------------------------
    # TAB 1: FIT SCORE
    # -------------------------------------------------------------------------
    with tab1:
        st.subheader("JD ↔ CV Fit Analysis")
        st.markdown(st.session_state.fit_score_result)
    
    # -------------------------------------------------------------------------
    # TAB 2: CANDIDATE SNAPSHOT
    # -------------------------------------------------------------------------
    with tab2:
        st.subheader("Structured Candidate Profile")
        
        if st.session_state.snapshot_json:
            data = st.session_state.snapshot_json
            
            # Display in columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📌 Basic Info**")
                st.write(f"**Name:** {data.get('name', 'N/A')}")
                st.write(f"**Email:** {data.get('email', 'N/A')}")
                st.write(f"**Phone:** {data.get('phone', 'N/A')}")
                st.write(f"**LinkedIn:** {data.get('linkedin', 'N/A')}")
                
                st.markdown("**💼 Current Position**")
                st.write(f"**Role:** {data.get('current_role', 'N/A')}")
                st.write(f"**Company:** {data.get('current_company', 'N/A')}")
                st.write(f"**Experience:** {data.get('experience_years', 'N/A')} years")
            
            with col2:
                st.markdown("**🛠️ Skills**")
                skills = data.get('skills', [])
                if skills:
                    st.write(", ".join(skills))
                else:
                    st.write("N/A")
                
                st.markdown("**🔧 Tools & Technologies**")
                tools = data.get('tools_technologies', [])
                if tools:
                    st.write(", ".join(tools))
                else:
                    st.write("N/A")
                
                st.markdown("**🎓 Education**")
                education = data.get('education', [])
                if education:
                    for edu in education:
                        st.write(f"• {edu.get('degree', '')} - {edu.get('institution', '')} ({edu.get('year', '')})")
                else:
                    st.write("N/A")
            
            st.markdown("**📝 Summary**")
            st.info(data.get('summary', 'N/A'))
            
            st.markdown("**🏆 Key Achievements**")
            achievements = data.get('key_achievements', [])
            if achievements:
                for ach in achievements:
                    st.write(f"• {ach}")
            else:
                st.write("N/A")
            
            st.divider()
            
            # Export options
            st.subheader("📥 Export Data")
            col1, col2 = st.columns(2)
            
            with col1:
                # JSON download
                json_str = json.dumps(data, indent=2, ensure_ascii=False)
                st.download_button(
                    label="⬇️ Download JSON",
                    data=json_str,
                    file_name="candidate_profile.json",
                    mime="application/json"
                )
            
            with col2:
                # CSV-friendly format
                csv_data = f"""Name,Email,Phone,Current Role,Company,Experience,Skills
"{data.get('name', '')}","{data.get('email', '')}","{data.get('phone', '')}","{data.get('current_role', '')}","{data.get('current_company', '')}","{data.get('experience_years', '')}","{', '.join(data.get('skills', []))}"
"""
                st.download_button(
                    label="⬇️ Download CSV",
                    data=csv_data,
                    file_name="candidate_profile.csv",
                    mime="text/csv"
                )
        
        else:
            # Show raw text if JSON parsing failed
            st.warning("Could not parse structured data. Showing raw response:")
            st.text(st.session_state.snapshot_result)
    
    # -------------------------------------------------------------------------
    # TAB 3: INTERVIEW KIT
    # -------------------------------------------------------------------------
    with tab3:
        st.subheader("Interview Preparation Kit")
        st.markdown(st.session_state.interview_kit_result)
        
        # Copy-friendly export
        st.divider()
        st.download_button(
            label="⬇️ Download Interview Kit",
            data=st.session_state.interview_kit_result,
            file_name="interview_kit.txt",
            mime="text/plain"
        )

else:
    # Show instructions when no analysis has been run
    st.info("👈 **Get started:** Upload a CV and paste a job description in the sidebar, then click 'Run Full Analysis'")
    
    # Feature highlights
    st.markdown("### What you'll get:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 📊 Fit Score")
        st.write("0-100 match score with evidence from the CV showing why the candidate fits (or doesn't).")
    
    with col2:
        st.markdown("#### 👤 Candidate Snapshot")
        st.write("Structured profile extraction: skills, experience, education — exportable as JSON/CSV.")
    
    with col3:
        st.markdown("#### 📝 Interview Kit")
        st.write("6 role-specific questions with rubrics + 2 clarification questions for gaps.")

# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.caption("CV Analyzer Pro • Built for recruiters who value quality over quantity • Undergraduate Project 2025")
