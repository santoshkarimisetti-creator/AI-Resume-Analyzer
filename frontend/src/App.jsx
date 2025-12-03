import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

export default function App() {
  const [resp, setResp] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);
  const [file,setFile]=useState(null);
  const [jd,setJd]=useState('');
  const [extractedText, setExtractedText] = useState("");
  const [analyzeLoading, setAnalyzeLoading] = useState(false);
  const [analyzeError, setAnalyzeError] = useState(null);


  const handleFileChange=(e)=>{
    const selectedFile=e.target.files[0];
    if(selectedFile){
      const validTypes=['application/pdf','application/msword','application/vnd.openxmlformats-officedocument.wordprocessingml.document'];
      if(validTypes.includes(selectedFile.type)){
        setFile(selectedFile);
      }else{
        alert('Invalid file type. Please upload a PDF or Word document.');
        e.target.value=null;
        setFile(null);
      }
    }
  }
  
  const analyzeResume=async()=>{
    if(!file){
      alert('Please upload a resume file first.');
      return;
    }
    setLoading(true);
    setErr(null);
    try{
      const formData=new FormData();
      formData.append('resume',file);
      if(jd &&jd.trim())
        formData.append('job_description',jd.trim());

      for (const pair of formData.entries()) {
        console.log('formData:', pair[0], pair[1]);
      }
      
      const { data } = await axios.post(
        "http://127.0.0.1:8000/api/analyzer/analyze/resume/",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
    );

      setResp(data);
    }catch(e){
      setErr(e.response?.data ?? e.message);
      setResp(null);
    }finally{
      setLoading(false);
    }
  };

  return (
  <div className="app-root">
    <div className="app-card">
      <h1 className="app-title">Resume Analyzer</h1>

      {/* --- File Input --- */}
      <div className="field-group">
        <label className="field-label">Upload Resume (.pdf or .docx):</label>
        <input
          type="file"
          accept=".pdf,.docx"
          onChange={handleFileChange}
          className="file-input"
        />
        {file && <p className="file-selected">Selected: {file.name}</p>}
      </div>

      {/* --- JD Input --- */}
      <div className="field-group">
        <label className="field-label">Job Description (optional):</label>
        <textarea
          value={jd}
          onChange={(e) => setJd(e.target.value)}
          placeholder="Paste job description here..."
          className="jd-textarea"
        />
      </div>

      <button
        onClick={analyzeResume}
        disabled={loading}
        className={`primary-btn ${loading ? "primary-btn-disabled" : ""}`}
      >
        {loading ? "Analyzing…" : "Analyze Resume"}
      </button>

      {/* --- ERROR MESSAGE --- */}
      {err && <p className="error-text">Error: {err}</p>}

      {/* --- RESULTS SECTION --- */}
      {resp && (
        <div className="result-card">
          {/* === MODE A: JD MATCH RESULTS === */}
          {resp.jd_present ? (
            <div>
              <div className="result-header">
                <span className="result-icon">🎯</span>
                <h2>JD Match Results</h2>
              </div>

              {/* Match Score */}
              <div className="score-block">
                <span
                  className={`score-value ${
                    resp.match_score >= 70 ? "score-good" : "score-medium"
                  }`}
                >
                  {resp.match_score}%
                </span>
                <p className="score-label">Match Score</p>
              </div>

              <div className="keywords-grid">
                {/* Matched Keywords */}
                <div className="keywords-column">
                  <h3 className="keywords-title matched-title">
                    ✅ Matched Keywords
                  </h3>
                  {resp.matched_keywords && resp.matched_keywords.length > 0 ? (
                    <div className="chip-list">
                      {resp.matched_keywords.map((kw, i) => (
                        <span key={i} className="chip chip-matched">
                          {kw}
                        </span>
                      ))}
                    </div>
                  ) : (
                    <p className="muted-text">No specific keywords matched.</p>
                  )}
                </div>

                {/* Missing Keywords */}
                <div className="keywords-column">
                  <h3 className="keywords-title missing-title">
                    ❌ Missing Keywords
                  </h3>
                  {resp.missing_keywords && resp.missing_keywords.length > 0 ? (
                    <div className="chip-list">
                      {resp.missing_keywords.map((kw, i) => (
                        <span key={i} className="chip chip-missing">
                          {kw}
                        </span>
                      ))}
                    </div>
                  ) : (
                    <p className="muted-text">No missing keywords found!</p>
                  )}
                </div>
              </div>
            </div>
          ) : (
            /* === MODE B: ATS READINESS RESULTS === */
            <div>
              <div className="result-header">
                <span className="result-icon">📋</span>
                <h2>ATS Readiness Report</h2>
              </div>

              {/* ATS Score */}
              <div className="score-block">
                <span
                  className={`score-value ${
                    resp.ats_score >= 70 ? "score-good" : "score-medium"
                  }`}
                >
                  {resp.ats_score}
                </span>
                <p className="score-label">ATS Readiness Score (0–100)</p>
              </div>

              {/* Breakdown Grid */}
              <div className="ats-grid">
                {/* Section Analysis */}
                <div className="ats-panel">
                  <h3 className="panel-title">📂 Section Analysis</h3>
                  <ul className="section-list">
                    {resp.section_analysis &&
                      Object.entries(resp.section_analysis).map(
                        ([section, present]) => (
                          <li key={section} className="section-row">
                            <span className="section-name">
                              {section.charAt(0).toUpperCase() +
                                section.slice(1)}
                            </span>
                            <span className="section-status">
                              {present ? "✅ Found" : "⚠️ Missing"}
                            </span>
                          </li>
                        )
                      )}
                  </ul>
                </div>

                {/* Detailed Scores */}
                <div className="ats-panel">
                  <h3 className="panel-title">📊 Scoring Details</h3>
                  <p>
                    <strong>Readability:</strong> {resp.readability_score} / 30
                  </p>
                  <p>
                    <strong>Keyword Density:</strong>{" "}
                    {resp.keyword_density_score} / 20
                  </p>
                  <p>
                    <strong>Formatting:</strong> {resp.formatting_score} / 10
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* --- Debug View (Extracted Text) --- */}
          <div className="debug-block">
            <details>
              <summary className="debug-summary">
                ▶ Show Raw Extracted Text (Debug)
              </summary>
              
              {/* 🔹 TASK 4: Copy Button */}
              <button
                onClick={() => {
                  navigator.clipboard.writeText(resp.extracted_text);
                  alert("Text copied to clipboard!");
                }}
                className="primary-btn"
                style={{ 
                  margin: "10px 0", 
                  fontSize: "0.8rem", 
                  padding: "6px 12px", 
                  cursor: "pointer",
                  display: "block" 
                }}
              >
                📋 Copy Extracted Text
              </button>

              <pre className="debug-pre">{resp.extracted_text}</pre>
            </details>
          </div>
        </div>
      )}
    </div>
  </div>
);
}
