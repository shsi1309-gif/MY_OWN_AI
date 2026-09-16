import React, { useState } from 'react';
import { Send, Sparkles, ShieldCheck, RefreshCw, Cpu, BookOpen, Layers, CheckCircle, AlertTriangle, ArrowRight } from 'lucide-react';

const SUGGESTED_QUESTIONS = [
  'What is gradient descent and how does learning rate affect it?',
  'Explain Modern Portfolio Theory and DCF valuation.',
  'How does the 3-tier architecture with Express Gateway work?',
  'What are the core steps in LangGraph Self-Correcting RAG?'
];

const NODE_INFO = {
  retrieve: { icon: '🔍', title: 'Vector Retrieval', color: 'var(--cs)' },
  grade_documents: { icon: '⚖️', title: 'Document Grader', color: 'var(--math)' },
  rewrite_query: { icon: '🔄', title: 'Query Rewriter', color: 'var(--food)' },
  generate: { icon: '⚡', title: 'Grounded Generator', color: 'var(--green)' },
  hallucination_check: { icon: '🛡️', title: 'Guardrail Verifier', color: 'var(--accent)' }
};

export default function AgenticRagChat({
  onAskRag,
  chatHistory = [],
  isThinking = false,
  onClearHistory = null
}) {
  const [question, setQuestion] = useState('');
  const [topK, setTopK] = useState(3);
  const [expandedDoc, setExpandedDoc] = useState(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!question.trim() || isThinking) return;
    onAskRag({
      question: question.trim(),
      k: topK
    });
    setQuestion('');
  };

  const handleSelectSuggestion = (q) => {
    setQuestion(q);
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
      {/* RAG Controls & Header */}
      <div>
        <div className="sec" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <span>LangGraph Self-Correcting RAG</span>
          {onClearHistory && chatHistory.length > 0 && (
            <button
              type="button"
              onClick={onClearHistory}
              className="del"
              style={{ padding: '1px 6px', fontSize: '9px' }}
            >
              Clear
            </button>
          )}
        </div>

        {/* Suggested Queries */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px', marginBottom: '10px' }}>
          {SUGGESTED_QUESTIONS.map((q, idx) => (
            <button
              key={idx}
              type="button"
              className="algo-btn"
              style={{
                textAlign: 'left',
                fontSize: '10px',
                padding: '6px 8px',
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis'
              }}
              onClick={() => handleSelectSuggestion(q)}
            >
              💡 {q}
            </button>
          ))}
        </div>

        {/* Input Form */}
        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
          <div>
            <input
              type="text"
              placeholder="Ask anything from your vector knowledge base..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              disabled={isThinking}
            />
          </div>

          <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
            <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: '6px', fontSize: '10px', color: 'var(--muted)' }}>
              <span>Context:</span>
              <input
                type="range"
                min="1"
                max="8"
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                style={{ flex: 1 }}
              />
              <span style={{ color: 'var(--cs)', fontWeight: 600 }}>{topK} Chunks</span>
            </div>

            <button
              type="submit"
              className="btn-p"
              style={{ width: 'auto', padding: '7px 14px' }}
              disabled={isThinking || !question.trim()}
            >
              <Send size={13} />
              {isThinking ? 'Reasoning...' : 'Ask AI'}
            </button>
          </div>
        </form>
      </div>

      {/* Thinking state indicator */}
      {isThinking && (
        <div className="thinking" style={{ padding: '12px', background: 'var(--bg)', borderRadius: '8px', border: '1px solid var(--border)' }}>
          <div className="spinner" />
          <span>LangGraph StateGraph is retrieving, grading, and generating grounded answer...</span>
        </div>
      )}

      {/* Chat Messages */}
      <div className="chat-history">
        {chatHistory.length === 0 && !isThinking && (
          <div style={{ fontSize: '11px', color: 'var(--muted)', textAlign: 'center', padding: '24px 12px', border: '1px dashed var(--border)', borderRadius: '8px' }}>
            <Sparkles size={24} color="var(--accent)" style={{ margin: '0 auto 8px', display: 'block', opacity: 0.6 }} />
            Enter a question above to trigger the <strong>LangGraph Self-Correcting RAG workflow</strong>.
            Documents will be retrieved from ChromaDB, verified for relevance, synthesized, and checked against hallucinations.
          </div>
        )}

        {chatHistory.map((msg, index) => (
          <div key={index} style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {/* User Question */}
            <div className="chat-q">
              <strong style={{ color: 'var(--accent)', marginRight: '6px' }}>User:</strong>
              {msg.question}
            </div>

            {/* LangGraph Trace Timeline */}
            {msg.trace && msg.trace.length > 0 && (
              <div className="trace-box">
                <div className="trace-header">
                  <span>
                    <Cpu size={12} style={{ marginRight: '4px', verticalAlign: 'middle' }} />
                    LangGraph StateGraph Execution Trace
                  </span>
                  <span className="trace-badge">{msg.trace.length} Steps</span>
                </div>

                {msg.trace.map((step, sIdx) => {
                  const info = NODE_INFO[step.node] || {
                    icon: '⚡',
                    title: step.node,
                    color: 'var(--accent)'
                  };

                  return (
                    <div key={sIdx} className={`trace-step ${step.node}`}>
                      <span className="trace-icon">{info.icon}</span>
                      <div className="trace-content">
                        <div className="trace-node">{info.title}</div>
                        <div className="trace-detail">{step.detail || step.message || JSON.stringify(step)}</div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* Assistant Grounded Answer */}
            <div className="chat-a">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
                <div className="chat-a-label">
                  <ShieldCheck size={12} style={{ marginRight: '4px', verticalAlign: 'middle' }} />
                  GROUNDED AI SYNTHESIS
                </div>
                {msg.grounded && (
                  <span
                    style={{
                      fontSize: '9px',
                      color: 'var(--green)',
                      background: 'rgba(166, 227, 161, 0.12)',
                      padding: '2px 6px',
                      borderRadius: '4px',
                      border: '1px solid rgba(166, 227, 161, 0.3)'
                    }}
                  >
                    ✓ 100% Guardrail Verified
                  </span>
                )}
              </div>

              <div className="chat-a-text">{msg.answer}</div>

              {/* Citations Context */}
              {msg.documents && msg.documents.length > 0 && (
                <div className="chat-ctx">
                  <div className="chat-ctx-label">
                    <BookOpen size={10} style={{ marginRight: '4px', verticalAlign: 'middle' }} />
                    RETRIEVED CITATIONS ({msg.documents.length}):
                  </div>
                  <div>
                    {msg.documents.map((doc, dIdx) => (
                      <span
                        key={dIdx}
                        className="ctx-chip"
                        onClick={() => setExpandedDoc(expandedDoc === dIdx ? null : dIdx)}
                        title="Click to toggle excerpt"
                      >
                        📄 {doc.metadata?.title || doc.title || `Chunk #${doc.id || dIdx + 1}`}
                        {doc.score !== undefined && ` (${(doc.score * 100).toFixed(0)}%)`}
                      </span>
                    ))}
                  </div>

                  {expandedDoc !== null && msg.documents[expandedDoc] && (
                    <div className="ctx-expand">
                      <strong style={{ color: 'var(--cs)' }}>
                        Excerpt from {msg.documents[expandedDoc].title || msg.documents[expandedDoc].metadata?.title || 'Document'}:
                      </strong>
                      <p style={{ marginTop: '4px' }}>
                        {msg.documents[expandedDoc].content || msg.documents[expandedDoc].metadata?.text || msg.documents[expandedDoc].text}
                      </p>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
