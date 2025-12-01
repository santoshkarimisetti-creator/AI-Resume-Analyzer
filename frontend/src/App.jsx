import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

export default function App() {
  const [resp, setResp] = useState(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState(null);

  const testBackend = async () => {
    setLoading(true);
    setErr(null);
    try {
      const { data } = await axios.get('http://127.0.0.1:8000/api/analyzer/test/');
      setResp(data);
    } catch (e) {
      setErr(e.response?.data ?? e.message);
      setResp(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="App" style={{ padding: 24, textAlign: 'center' }}>
      <h1>Resume Analyzer</h1>
      <button onClick={testBackend} disabled={loading} style={{ padding: '8px 16px' }}>
        {loading ? 'Testing…' : 'Test Backend'}
      </button>

      {resp && (
        <pre style={{ marginTop: 16, textAlign: 'left', display: 'inline-block' }}>
          {JSON.stringify(resp, null, 2)}
        </pre>
      )}

      {err && <p style={{ color: 'red', marginTop: 12 }}>Error: {JSON.stringify(err)}</p>}
    </div>
  );
}
