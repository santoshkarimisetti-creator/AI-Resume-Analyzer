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
    <div className="App" style={{ padding: 24, textAlign: 'center', maxWidth: 600, margin: '0 auto' }}>
      <h1>Resume Analyzer</h1>

      <div style={{ marginBottom: 16, textAlign: 'left' }}>
        <label>Upload Resume (.pdf or .docx):</label>
        <input type="file" accept=".pdf,.docx" onChange={handleFileChange} style={{ display: 'block', marginTop: 8 }} />
        {file && <p style={{ color: 'green', marginTop: 8 }}>Selected: {file.name}</p>}
      </div>

      <div style={{ marginBottom: 16, textAlign: 'left' }}>
        <label>Job Description (optional):</label>
        <textarea
          value={jd}
          onChange={(e) => setJd(e.target.value)}
          placeholder="Paste job description here..."
          style={{ width: '100%', height: 120, marginTop: 8, padding: 8, fontFamily: 'monospace' }}
        />
      </div>

      <button onClick={analyzeResume} disabled={loading} style={{ padding: '10px 20px', fontSize: 16 }}>
        {loading ? 'Analyzing…' : 'Analyze Resume'}
      </button>

      
      {resp && resp.extracted_text && (
  <div style={{ marginTop: 24, textAlign: 'left' }}>
    <h3>Extracted Resume Text (Debug View)</h3>
    <pre
      style={{
        whiteSpace: 'pre-wrap',
        background: '#222',      // dark background
        color: '#fff',           // white text
        padding: 12,
        borderRadius: 4,
        maxHeight: 300,
        overflowY: 'auto',
        fontFamily: 'monospace',
      }}
    >
      {resp.extracted_text}
    </pre>
  </div>
)}



      {err && <p style={{ color: 'red', marginTop: 12 }}>Error: {err}</p>}
    </div>
  );
}
