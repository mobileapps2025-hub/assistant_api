# MCL Assistant API - Frontend Integration Changes

## Overview

The MCL Assistant API has been enhanced with **source attribution** and **document chunk tracking**. This document outlines the changes needed in the frontend to take advantage of these new features.

## 🆕 New API Features

### 1. Source Attribution in Responses
- Responses now include specific document sources
- Each answer shows which MCL documents were used
- Chunk-level precision for better traceability

### 2. New Endpoints for Enhanced Functionality
- `/api/chunks` - Get document chunk information
- `/api/search` - Search for relevant chunks
- Enhanced `/health` endpoint with chunk count

### 3. Enhanced Response Format
Responses now include source information at the end of each answer.

## 🔄 Required Frontend Changes

### 1. Update Response Parsing

**Before:**
```typescript
interface ChatResponse {
  messages: Message[]
}

interface Message {
  role: string
  content: ContentItem[]
}
```

**After (Enhanced):**
```typescript
interface ChatResponse {
  messages: Message[]
}

interface Message {
  role: string
  content: ContentItem[]
  sources?: string[] // New: extracted source information
}

interface ContentItem {
  text: string
  type: string
}
```

### 2. Source Extraction Utility

Add a utility function to extract and parse source information from responses:

```typescript
function extractSourcesFromResponse(content: string): { cleanContent: string, sources: string[] } {
  const sourcesMatch = content.match(/📚 \*\*Sources:\*\*\n((?:• .+\n?)+)/);
  
  if (sourcesMatch) {
    const sourcesText = sourcesMatch[1];
    const sources = sourcesText
      .split('\n')
      .map(line => line.replace(/^• /, '').trim())
      .filter(line => line.length > 0);
    
    const cleanContent = content.replace(/📚 \*\*Sources:\*\*\n((?:• .+\n?)+)/, '').trim();
    
    return { cleanContent, sources };
  }
  
  return { cleanContent: content, sources: [] };
}
```

### 3. Enhanced Message Display Component

```typescript
interface MessageDisplayProps {
  message: Message
}

const MessageDisplay: React.FC<MessageDisplayProps> = ({ message }) => {
  const [showSources, setShowSources] = useState(false)
  
  // Extract content and sources
  const content = message.content?.[0]?.text || ''
  const { cleanContent, sources } = extractSourcesFromResponse(content)
  
  return (
    <div className={`message ${message.role}`}>
      <div className="message-content">
        {/* Render the clean content without sources */}
        <ReactMarkdown>{cleanContent}</ReactMarkdown>
        
        {/* Show sources if available */}
        {sources.length > 0 && (
          <div className="message-sources">
            <button 
              onClick={() => setShowSources(!showSources)}
              className="sources-toggle"
            >
              📚 Sources ({sources.length}) {showSources ? '▼' : '▶'}
            </button>
            
            {showSources && (
              <div className="sources-list">
                {sources.map((source, index) => (
                  <div key={index} className="source-item">
                    <span className="source-icon">📄</span>
                    <span className="source-text">{source}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
```

### 4. Enhanced CSS Styles

Add these styles for the new source display:

```css
.message-sources {
  margin-top: 12px;
  border-top: 1px solid #e0e0e0;
  padding-top: 8px;
}

.sources-toggle {
  background: #f5f5f5;
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 6px 12px;
  cursor: pointer;
  font-size: 12px;
  color: #666;
  transition: background-color 0.2s;
}

.sources-toggle:hover {
  background: #e9e9e9;
}

.sources-list {
  margin-top: 8px;
  padding: 8px;
  background: #f9f9f9;
  border-radius: 4px;
  border: 1px solid #e0e0e0;
}

.source-item {
  display: flex;
  align-items: center;
  margin-bottom: 4px;
  font-size: 12px;
  color: #555;
}

.source-icon {
  margin-right: 6px;
}

.source-text {
  font-family: 'Monaco', 'Consolas', monospace;
  background: #fff;
  padding: 2px 6px;
  border-radius: 3px;
  border: 1px solid #ddd;
}
```

### 5. New Debug Panel (Optional)

Add a debug panel to view available document chunks:

```typescript
const DebugPanel: React.FC = () => {
  const [chunks, setChunks] = useState<any>(null)
  const [searchQuery, setSearchQuery] = useState('')
  const [searchResults, setSearchResults] = useState<any>(null)
  
  const fetchChunks = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/chunks')
      const data = await response.json()
      setChunks(data)
    } catch (error) {
      console.error('Error fetching chunks:', error)
    }
  }
  
  const searchChunks = async () => {
    if (!searchQuery.trim()) return
    
    try {
      const response = await fetch('http://localhost:8000/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: searchQuery, max_results: 5 })
      })
      const data = await response.json()
      setSearchResults(data)
    } catch (error) {
      console.error('Error searching chunks:', error)
    }
  }
  
  return (
    <div className="debug-panel">
      <h3>MCL Knowledge Base Debug</h3>
      
      <div className="debug-section">
        <button onClick={fetchChunks}>Load Document Chunks</button>
        {chunks && (
          <div>
            <p>Total Documents: {chunks.total_documents}</p>
            <p>Total Chunks: {chunks.total_chunks}</p>
          </div>
        )}
      </div>
      
      <div className="debug-section">
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search chunks..."
        />
        <button onClick={searchChunks}>Search</button>
        
        {searchResults && (
          <div>
            <p>Found {searchResults.total_results} results</p>
            {searchResults.results.map((result: any, index: number) => (
              <div key={index} className="search-result">
                <strong>{result.document_name}</strong> (Chunk {result.chunk_index + 1})
                <p>{result.content_preview}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
```

## 📊 Enhanced Health Check

The health endpoint now provides more information:

```typescript
interface HealthResponse {
  status: string
  knowledge_base_loaded: boolean
  vector_store_id: string
  total_document_chunks: number // New field
}

const checkHealth = async (): Promise<HealthResponse> => {
  const response = await fetch('http://localhost:8000/health')
  return await response.json()
}
```

## 🎯 Recommended UI Improvements

### 1. Source Confidence Indicator
```typescript
const getSourceConfidence = (sources: string[]): 'high' | 'medium' | 'low' => {
  if (sources.length >= 3) return 'high'
  if (sources.length >= 2) return 'medium'
  return 'low'
}

// Use in UI:
<div className={`confidence-indicator ${getSourceConfidence(sources)}`}>
  {sources.length} source{sources.length !== 1 ? 's' : ''}
</div>
```

### 2. Document Type Icons
```typescript
const getDocumentIcon = (documentName: string): string => {
  if (documentName.includes('How-to')) return '📖'
  if (documentName.includes('Guide')) return '📋'
  if (documentName.includes('Release')) return '🔄'
  if (documentName.includes('Common Mistakes')) return '⚠️'
  if (documentName.includes('Business')) return '💼'
  return '📄'
}
```

### 3. Enhanced Loading States
```typescript
const ChatInterface: React.FC = () => {
  const [isSearching, setIsSearching] = useState(false)
  
  const sendMessage = async (text: string) => {
    setIsSearching(true)
    // ... existing logic
    setIsSearching(false)
  }
  
  return (
    <div>
      {isSearching && (
        <div className="searching-indicator">
          🔍 Searching MCL documentation...
        </div>
      )}
      {/* ... rest of component */}
    </div>
  )
}
```

## 🔧 Migration Steps

### Step 1: Update API Client
1. Update TypeScript interfaces to include source information
2. Add the `extractSourcesFromResponse` utility function
3. Update the health check to handle the new response format

### Step 2: Update Message Display
1. Modify message components to extract and display sources
2. Add CSS styles for source display
3. Add source toggle functionality

### Step 3: Optional Enhancements
1. Add debug panel for development/testing
2. Implement source confidence indicators
3. Add document type icons
4. Enhance loading states

### Step 4: Testing
1. Test source extraction with various response formats
2. Verify source display toggle works correctly
3. Test debug endpoints if implemented
4. Verify graceful handling when no sources are available

## ⚠️ Important Notes

### Backward Compatibility
- The API changes are **backward compatible**
- Existing frontend code will continue to work
- Source information is additive - responses without sources will work normally

### Performance Considerations
- Source extraction is done client-side
- No additional API calls required for basic functionality
- Debug endpoints are optional and should be used sparingly in production

### Styling Flexibility
- Source display can be customized to match your design system
- Icons and colors can be changed to fit your brand
- Source toggle can be replaced with always-visible sources if preferred

## 🚀 Benefits of These Changes

1. **Improved Trust**: Users can see exactly where information comes from
2. **Better Debugging**: Developers can trace response sources
3. **Enhanced UX**: Clear indication of information reliability
4. **Maintainability**: Easy to identify outdated or missing documentation

## 📝 Example Usage

After implementing these changes, your chat interface will show:

```
User: "How do I create a checklist in MCL?"
