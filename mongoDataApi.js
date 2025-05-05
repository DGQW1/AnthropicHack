// MongoDB Data API integration for PDF access
const fetch = (...args) => import('node-fetch').then(({default: fetch}) => fetch(...args));
require('dotenv').config();

// MongoDB Data API endpoint and key
const MONGODB_API_URL = process.env.MONGODB_API_URL || "https://data.mongodb-api.com/app/data-api/endpoint/data/v1/action/find";
const MONGODB_API_KEY = process.env.MONGODB_API_KEY;
const DATA_SOURCE = process.env.MONGODB_DATA_SOURCE || "AnthropicHack";
const DATABASE = process.env.MONGODB_DATABASE || "pdfs_db";
const FILES_COLLECTION = "fs.files"; // GridFS files collection

/**
 * List all files in the database
 * @returns {Promise<Array>} Array of file metadata
 */
async function listAllFiles() {
  try {
    const response = await fetch(MONGODB_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Request-Headers': '*',
        'api-key': MONGODB_API_KEY
      },
      body: JSON.stringify({
        "dataSource": DATA_SOURCE,
        "database": DATABASE,
        "collection": FILES_COLLECTION,
        "projection": {
          "_id": 1,
          "filename": 1,
          "contentType": 1,
          "length": 1,
          "uploadDate": 1,
          "metadata": 1
        }
      })
    });

    if (!response.ok) {
      throw new Error(`API request failed with status ${response.status}: ${response.statusText}`);
    }

    const result = await response.json();
    return result.documents || [];
  } catch (error) {
    console.error('Error fetching files from MongoDB Data API:', error);
    throw error;
  }
}

/**
 * Find files for a specific homework number
 * @param {string} hwNumber Homework number
 * @returns {Promise<Array>} Array of matching file metadata
 */
async function findHomeworkQuestions(hwNumber) {
  const prefix = `270HW${hwNumber}_question`;
  
  try {
    const response = await fetch(MONGODB_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Request-Headers': '*',
        'api-key': MONGODB_API_KEY
      },
      body: JSON.stringify({
        "dataSource": DATA_SOURCE,
        "database": DATABASE,
        "collection": FILES_COLLECTION,
        "filter": {
          "filename": { "$regex": `^${prefix}` }
        },
        "projection": {
          "_id": 1,
          "filename": 1,
          "contentType": 1,
          "length": 1,
          "metadata": 1
        }
      })
    });

    if (!response.ok) {
      throw new Error(`API request failed with status ${response.status}: ${response.statusText}`);
    }

    const result = await response.json();
    return result.documents || [];
  } catch (error) {
    console.error(`Error fetching homework ${hwNumber} from MongoDB Data API:`, error);
    throw error;
  }
}

/**
 * Find a specific file by filename
 * @param {string} filename Filename to find
 * @returns {Promise<Object>} File metadata
 */
async function findFileByName(filename) {
  try {
    const response = await fetch(MONGODB_API_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Request-Headers': '*',
        'api-key': MONGODB_API_KEY
      },
      body: JSON.stringify({
        "dataSource": DATA_SOURCE,
        "database": DATABASE,
        "collection": FILES_COLLECTION,
        "filter": {
          "filename": filename
        }
      })
    });

    if (!response.ok) {
      throw new Error(`API request failed with status ${response.status}: ${response.statusText}`);
    }

    const result = await response.json();
    return result.documents && result.documents.length > 0 ? result.documents[0] : null;
  } catch (error) {
    console.error(`Error finding file ${filename} from MongoDB Data API:`, error);
    throw error;
  }
}

// Note: Actual file content retrieval requires a different approach since
// GridFS chunks need to be reassembled. For actual file content, you might need
// to create a separate API endpoint that uses the MongoDB driver.

module.exports = {
  listAllFiles,
  findHomeworkQuestions,
  findFileByName
}; 