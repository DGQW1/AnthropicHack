// Frontend example for accessing MongoDB using Data API
// Save this as a separate file to include in your frontend application

// You'll need to store your MongoDB API key securely 
// For production, use environment variables or a secure backend proxy

// Base URL for your backend server
const API_BASE_URL = "http://localhost:3001";

/**
 * Get a list of all files in the database
 */
async function getAllFiles() {
  try {
    const response = await fetch(`${API_BASE_URL}/api/files`);
    if (!response.ok) {
      throw new Error(`Failed to fetch files: ${response.status} ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error("Error getting files:", error);
    throw error;
  }
}

/**
 * Get questions for a specific homework assignment
 * @param {string} hwNumber - Homework number
 */
async function getHomeworkQuestions(hwNumber) {
  try {
    const response = await fetch(`${API_BASE_URL}/api/homework/${hwNumber}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch homework ${hwNumber}: ${response.status} ${response.statusText}`);
    }
    return await response.json();
  } catch (error) {
    console.error(`Error getting homework ${hwNumber}:`, error);
    throw error;
  }
}

/**
 * Get direct file download URL
 * @param {string} filename - Filename to download
 */
function getFileUrl(filename) {
  return `${API_BASE_URL}/api/file/${filename}`;
}

/**
 * Example: List all files and display them
 */
async function displayAllFiles() {
  try {
    const filesResult = await getAllFiles();
    
    // Example: Update UI with files
    const filesList = document.getElementById('files-list');
    if (!filesList) return;
    
    filesList.innerHTML = '';
    
    if (filesResult.files && filesResult.files.length > 0) {
      filesResult.files.forEach(file => {
        const fileItem = document.createElement('div');
        fileItem.classList.add('file-item');
        
        const fileLink = document.createElement('a');
        fileLink.href = getFileUrl(file.filename);
        fileLink.textContent = file.filename;
        fileLink.target = '_blank';
        
        const fileInfo = document.createElement('span');
        fileInfo.textContent = ` (${formatFileSize(file.size)}, uploaded ${new Date(file.uploadDate).toLocaleString()})`;
        
        fileItem.appendChild(fileLink);
        fileItem.appendChild(fileInfo);
        filesList.appendChild(fileItem);
      });
    } else {
      filesList.innerHTML = '<p>No files found</p>';
    }
  } catch (error) {
    console.error("Error displaying files:", error);
    // Handle error in UI
  }
}

/**
 * Example: Load and display a specific homework's questions
 * @param {string} hwNumber - Homework number to display
 */
async function displayHomeworkQuestions(hwNumber) {
  try {
    const homeworkResult = await getHomeworkQuestions(hwNumber);
    
    // Example: Update UI with homework questions
    const questionsList = document.getElementById('questions-list');
    if (!questionsList) return;
    
    questionsList.innerHTML = '';
    
    if (homeworkResult.questions && homeworkResult.questions.length > 0) {
      const heading = document.createElement('h3');
      heading.textContent = `Homework #${hwNumber} - ${homeworkResult.questions.length} questions`;
      questionsList.appendChild(heading);
      
      homeworkResult.questions.forEach(question => {
        const questionItem = document.createElement('div');
        questionItem.classList.add('question-item');
        
        const questionTitle = document.createElement('h4');
        questionTitle.textContent = `Question ${question.questionNumber}`;
        
        const viewLink = document.createElement('a');
        viewLink.href = getFileUrl(question.filename);
        viewLink.textContent = 'View Question';
        viewLink.target = '_blank';
        viewLink.classList.add('btn', 'btn-primary');
        
        questionItem.appendChild(questionTitle);
        questionItem.appendChild(viewLink);
        questionsList.appendChild(questionItem);
      });
    } else {
      questionsList.innerHTML = `<p>No questions found for Homework #${hwNumber}</p>`;
    }
  } catch (error) {
    console.error(`Error displaying homework ${hwNumber}:`, error);
    // Handle error in UI
  }
}

/**
 * Helper: Format file size in bytes to human-readable format
 * @param {number} bytes - File size in bytes
 * @returns {string} Formatted file size
 */
function formatFileSize(bytes) {
  if (bytes === 0) return '0 Bytes';
  
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Example event listeners
document.addEventListener('DOMContentLoaded', () => {
  // Load all files when page loads
  displayAllFiles();
  
  // Setup homework selector
  const hwSelector = document.getElementById('hw-selector');
  if (hwSelector) {
    hwSelector.addEventListener('change', (event) => {
      const hwNumber = event.target.value;
      if (hwNumber) {
        displayHomeworkQuestions(hwNumber);
      }
    });
  }
}); 