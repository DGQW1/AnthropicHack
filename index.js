console.log('🛠️  index.js loaded, setting up routes…');


require('dotenv').config();
const express = require('express');
const cors    = require('cors');
const multer  = require('multer');
const fs      = require('fs');
const { MongoClient, GridFSBucket } = require('mongodb');
const Grid    = require('gridfs-stream');
const path = require('path');
const mongoDataApi = require('./mongoDataApi');

const app    = express();
app.use(cors());                   // enable CORS for frontend access
const upload = multer({ dest: 'uploads/' });

// Serve static files (HTML, CSS, JS) from the root directory
app.use(express.static(__dirname));
console.log('Serving static files from:', __dirname);

// Debug routes
console.log('Setting up routes...');

// Serve the test page - original route
app.get('/test', (req, res) => {
    console.log('TEST route accessed!');
    res.send(`
    <!DOCTYPE html>
    <html>
    <head>
        <title>PDF Splitter Test</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            .form-container {
                border: 1px solid #ccc;
                padding: 20px;
                border-radius: 5px;
            }
            .result {
                margin-top: 20px;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                white-space: pre-wrap;
            }
        </style>
    </head>
    <body>
        <h1>PDF Splitter Test</h1>
        <div class="form-container">
            <form id="uploadForm">
                <input type="file" id="pdfFile" accept=".pdf" required>
                <button type="submit">Upload and Split</button>
            </form>
        </div>
        <div id="result" class="result"></div>

        <script>
            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const resultDiv = document.getElementById('result');
                const fileInput = document.getElementById('pdfFile');
                
                if (!fileInput.files.length) {
                    resultDiv.textContent = 'Please select a file';
                    return;
                }

                const formData = new FormData();
                formData.append('file', fileInput.files[0]);

                try {
                    resultDiv.textContent = 'Uploading...';
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    resultDiv.textContent = JSON.stringify(result, null, 2);
                    
                    if (result.files) {
                        resultDiv.textContent += '\\n\\nDownload links:\\n';
                        result.files.forEach(file => {
                            resultDiv.textContent += \`\\nhttp://localhost:3001/file/\${file.filename}\`;
                        });
                    }
                } catch (error) {
                    resultDiv.textContent = 'Error: ' + error.message;
                }
            });
        </script>
    </body>
    </html>
    `);
});

// Serve the question viewer page
app.get('/viewer', (req, res) => {
    console.log('Question viewer accessed');
    const viewerPath = path.resolve(__dirname, 'question-viewer.html');
    console.log('Serving viewer from path:', viewerPath);
    res.setHeader('Content-Type', 'text/html');
    res.sendFile(viewerPath);
});

// Simple viewer for testing routes
app.get('/viewer-simple', (req, res) => {
    console.log('Simple viewer accessed');
    const simplePath = path.resolve(__dirname, 'simple-viewer.html');
    console.log('Serving simple viewer from path:', simplePath);
    res.setHeader('Content-Type', 'text/html');
    res.sendFile(simplePath);
});

// Also add a direct HTML response route
app.get('/viewer-direct', (req, res) => {
    console.log('Direct HTML viewer accessed');
    res.setHeader('Content-Type', 'text/html');
    res.send(`
        <!DOCTYPE html>
        <html>
        <head>
            <title>Direct HTML Response</title>
        </head>
        <body>
            <h1>Direct HTML Response Works!</h1>
            <p>This HTML is sent directly in the response, not from a file.</p>
            <p><a href="/test">Go to Test Upload Page</a></p>
        </body>
        </html>
    `);
});

// Alternative route in case /test has issues
app.get('/uploader', (req, res) => {
    console.log('UPLOADER route accessed!');
    res.send(`
    <!DOCTYPE html>
    <html>
    <head>
        <title>PDF Uploader</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            .form-container {
                border: 1px solid #ccc;
                padding: 20px;
                border-radius: 5px;
            }
            .result {
                margin-top: 20px;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                white-space: pre-wrap;
            }
        </style>
    </head>
    <body>
        <h1>PDF Uploader</h1>
        <div class="form-container">
            <form id="uploadForm">
                <input type="file" id="pdfFile" accept=".pdf" required>
                <button type="submit">Upload and Split</button>
            </form>
        </div>
        <div id="result" class="result"></div>

        <script>
            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                const resultDiv = document.getElementById('result');
                const fileInput = document.getElementById('pdfFile');
                
                if (!fileInput.files.length) {
                    resultDiv.textContent = 'Please select a file';
                    return;
                }

                const formData = new FormData();
                formData.append('file', fileInput.files[0]);

                try {
                    resultDiv.textContent = 'Uploading...';
                    const response = await fetch('/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const result = await response.json();
                    resultDiv.textContent = JSON.stringify(result, null, 2);
                    
                    if (result.files) {
                        resultDiv.textContent += '\\n\\nDownload links:\\n';
                        result.files.forEach(file => {
                            resultDiv.textContent += \`\\nhttp://localhost:3000/file/\${file.filename}\`;
                        });
                    }
                } catch (error) {
                    resultDiv.textContent = 'Error: ' + error.message;
                }
            });
        </script>
    </body>
    </html>
    `);
});

// Initialize defaults for MongoDB-related variables
let gfs, gfsBucket;
let mongoConnected = false;

// Use a more resilient MongoDB connection approach
try {
  // Try to connect to MongoDB if URI is provided
  if (process.env.MONGO_URI) {
    console.log('Attempting to connect to MongoDB...');
    MongoClient.connect(process.env.MONGO_URI)
      .then(client => {
        const db = client.db();
        gfsBucket = new GridFSBucket(db, { bucketName: 'pdfs' });
        gfs = Grid(db, MongoClient);
        gfs.collection('pdfs');
        mongoConnected = true;
        console.log('✅ Connected to MongoDB');
      })
      .catch(err => {
        console.error('❌ MongoDB connection error:', err);
        console.log('⚠️ Running with limited functionality (database features disabled)');
        // Set up empty mock functions for MongoDB-related functionality
        setupMocks();
      });
  } else {
    console.log('❗ No MongoDB URI provided in environment variables');
    console.log('⚠️ Running with limited functionality (database features disabled)');
    // Set up empty mock functions for MongoDB-related functionality
    setupMocks();
  }
} catch (error) {
  console.error('❌ Error in MongoDB setup:', error);
  console.log('⚠️ Running with limited functionality (database features disabled)');
  // Set up empty mock functions for MongoDB-related functionality
  setupMocks();
}

// Set up mock functions when MongoDB is not available
function setupMocks() {
  // Create empty bucket with mock functions
  gfsBucket = {
    find: () => ({ 
      toArray: async () => [] 
    }),
    openUploadStream: () => ({
      id: 'mock-id',
      on: () => {}
    }),
    openDownloadStream: () => ({
      on: (event, callback) => {
        if (event === 'error') {
          callback(new Error('MongoDB not connected'));
        }
        return { pipe: () => {} };
      },
      pipe: () => {}
    })
  };
  
  gfs = {
    collection: () => {}
  };
}

// ─── 2) Health check ─────────────────────────────────────────────────────────
app.get('/', (_req, res) => res.send('Server is up'));

// Add a MongoDB connection status endpoint
app.get('/status', (_req, res) => {
  res.json({
    server: 'up',
    mongodb: mongoConnected ? 'connected' : 'disconnected',
    environment: process.env.NODE_ENV || 'development'
  });
});

// ─── 3) Upload endpoint ───────────────────────────────────────────────────────
app.post('/upload', upload.single('file'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'No file uploaded.' });
  }

  const { originalname, mimetype, path: tempPath } = req.file;
  console.log('Received file:', { originalname, mimetype, tempPath });
  
  try {
    // Simple file upload without PDF splitting
    const uploadStream = gfsBucket.openUploadStream(originalname, {
      contentType: mimetype,
      metadata: { uploadedAt: new Date() }
    });
    const fileId = uploadStream.id;

    fs.createReadStream(tempPath)
      .pipe(uploadStream)
      .on('error', err => {
        console.error('🔴 GridFS upload error:', err);
        res.sendStatus(500);
      })
      .on('finish', () => {
        fs.unlink(tempPath, unlinkErr => {
          if (unlinkErr) console.error('Temp file cleanup error:', unlinkErr);
        });
        res.json({ fileId, filename: originalname });
      });
  } catch (error) {
    console.error('Error processing file:', error);
    console.error('Error details:', {
      name: error.name,
      message: error.message,
      stack: error.stack
    });
    res.status(500).json({ 
      error: 'Error processing file',
      details: error.message
    });
  }
});

// ─── 4) List metadata ─────────────────────────────────────────────────────────────
// Add a debug route to list all files in GridFS
app.get('/files', async (req, res) => {
  console.log('Getting list of all files');
  
  try {
    const files = await gfsBucket.find({}).toArray();
    
    const fileInfo = files.map(file => ({
      id: file._id.toString(),
      filename: file.filename,
      contentType: file.contentType,
      size: file.length,
      uploadDate: file.uploadDate,
      metadata: file.metadata
    }));
    
    res.json({ count: fileInfo.length, files: fileInfo });
  } catch (error) {
    console.error('Error listing files:', error);
    res.status(500).json({ error: 'Could not list files' });
  }
});

// Map the /api/files endpoint to the existing /files endpoint
app.get('/api/files', async (req, res) => {
  console.log('API: Getting list of all files via existing endpoint');
  
  try {
    const files = await gfsBucket.find({}).toArray();
    
    const fileInfo = files.map(file => ({
      id: file._id.toString(),
      filename: file.filename,
      contentType: file.contentType,
      size: file.length,
      uploadDate: file.uploadDate,
      metadata: file.metadata
    }));
    
    res.json({ count: fileInfo.length, files: fileInfo });
  } catch (error) {
    console.error('Error listing files via API:', error);
    res.status(500).json({ error: 'Could not list files' });
  }
});

// Add a route to get questions from a specific homework file
app.get('/homework/:hwNumber', async (req, res) => {
  const hwNumber = req.params.hwNumber;
  const prefix = `270HW${hwNumber}_question`;
  
  console.log(`Looking for questions in HW${hwNumber}`);
  
  try {
    // Find all files that match the homework prefix
    const files = await gfsBucket.find({
      filename: { $regex: `^${prefix}` }
    }).toArray();
    
    if (!files || files.length === 0) {
      return res.status(404).json({ error: `No questions found for HW${hwNumber}` });
    }
    
    // Return info about the questions
    const questions = files.map(file => ({
      id: file._id.toString(),
      filename: file.filename,
      questionNumber: file.metadata?.questionNumber || 0,
      size: file.length,
      downloadUrl: `/file/${file.filename}`
    }));
    
    // Sort by question number
    questions.sort((a, b) => a.questionNumber - b.questionNumber);
    
    res.json({
      hwNumber,
      questionCount: questions.length,
      questions
    });
  } catch (error) {
    console.error(`Error listing questions for HW${hwNumber}:`, error);
    res.status(500).json({ error: 'Could not list questions' });
  }
});

// API route to get homework questions
app.get('/api/homework/:hwNumber', async (req, res) => {
  const hwNumber = req.params.hwNumber;
  
  console.log(`API: Looking for questions in HW${hwNumber}`);
  
  try {
    // Find all files that match the homework prefix
    const prefix = `270HW${hwNumber}_question`;
    const files = await gfsBucket.find({
      filename: { $regex: `^${prefix}` }
    }).toArray();
    
    if (!files || files.length === 0) {
      return res.status(404).json({ error: `No questions found for HW${hwNumber}` });
    }
    
    // Return info about the questions
    const questions = files.map(file => ({
      id: file._id.toString(),
      filename: file.filename,
      questionNumber: file.metadata?.questionNumber || 0,
      size: file.length,
      downloadUrl: `/api/file/${file.filename}`
    }));
    
    // Sort by question number
    questions.sort((a, b) => a.questionNumber - b.questionNumber);
    
    res.json({
      hwNumber,
      questionCount: questions.length,
      questions
    });
  } catch (error) {
    console.error(`Error listing questions for HW${hwNumber} via API:`, error);
    res.status(500).json({ error: 'Could not list questions' });
  }
});

// API route to get file metadata
app.get('/api/file-info/:filename', async (req, res) => {
  const filename = req.params.filename;
  console.log('API: Getting file info for', filename);
  
  try {
    const files = await gfsBucket.find({ filename }).toArray();
    
    if (!files || files.length === 0) {
      return res.status(404).json({ error: 'File not found' });
    }
    
    const file = files[0];
    
    res.json({
      id: file._id.toString(),
      filename: file.filename,
      contentType: file.contentType,
      size: file.length,
      uploadDate: file.uploadDate,
      metadata: file.metadata,
      downloadUrl: `/api/file/${file.filename}`
    });
  } catch (error) {
    console.error('Error getting file info via API:', error);
    res.status(500).json({ error: 'Could not get file info' });
  }
});

// For file content route
app.get('/api/file/:filename', async (req, res) => {
  const filename = req.params.filename;
  console.log('API: Download requested for', filename);

  try {
    // Find the file in GridFS
    const files = await gfsBucket.find({ filename }).toArray();
    
    if (!files || files.length === 0) {
      console.warn('⚠️ File not found in GridFS:', filename);
      return res.status(404).send('Not found');
    }
    
    const file = files[0];
    console.log('✔️ Found file in DB:', file.filename, 'size=', file.length);

    // Set response headers
    res.setHeader('Content-Type', file.contentType);
    res.setHeader('Content-Length', file.length);
    res.setHeader('Content-Disposition', `inline; filename="${file.filename}"`);

    // Stream the file data
    const downloadStream = gfsBucket.openDownloadStream(file._id);
    downloadStream
      .on('error', err => {
        console.error('🔴 GridFS download error:', err);
        if (!res.headersSent) res.sendStatus(500);
        else res.end();
      })
      .pipe(res);
  } catch (err) {
    console.error('❌ Error in API download handler:', err);
    if (!res.headersSent) res.sendStatus(500);
    else res.end();
  }
});

// ─── Download & stream a file by filename ────────────────────────────────────
app.get('/file/:filename', async (req, res) => {
    const filename = req.params.filename;
    console.log('📥 Download requested for', filename);
  
    try {
      // 1) Find the file document via the bucket
      const files = await gfsBucket.find({ filename }).toArray();
      if (!files || files.length === 0) {
        console.warn('⚠️  File not found in DB:', filename);
        return res.status(404).send('Not found');
      }
      const file = files[0];
      console.log('✔️  Found file in DB:', file.filename, 'size=', file.length);
  
      // 2) Set the headers up front
      res.setHeader('Content-Type',        file.contentType);
      res.setHeader('Content-Length',      file.length);
      res.setHeader(
        'Content-Disposition',
        `inline; filename="${file.filename}"`
      );
  
      // 3) Open the download stream and pipe it directly to res
      const downloadStream = gfsBucket.openDownloadStream(file._id);
      downloadStream
        .on('error', err => {
          console.error('🔴 GridFS download error:', err);
          // If headers already sent, just close; otherwise send 500
          if (!res.headersSent) res.sendStatus(500);
          else res.end();
        })
        .pipe(res);  // pipe will end res for you once the stream ends
  
    } catch (err) {
      console.error('❌ Error in download handler:', err);
      if (!res.headersSent) res.sendStatus(500);
      else res.end();
    }
  });

  
// ─── 6) Start the server ───────────────────────────────────────────────────────
const PORT = 3001; // Force port 3001 without checking process.env.PORT
app.listen(PORT, () => console.log(`🚀 Listening on http://localhost:${PORT}`));