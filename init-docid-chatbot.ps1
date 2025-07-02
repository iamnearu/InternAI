# Set project root path
$root = "C:\Users\Iamnearu\Documents\ThucTapAI\InternAI"

# Define folders to create
$folders = @(
    "$root\data\sample_images",
    "$root\ocr",
    "$root\caption",
    "$root\retriever",
    "$root\chatbot\nodes",
    "$root\app",
    "$root\frontend",
    "$root\utils",
    "$root\docker",
    "$root\notebooks"
)

# Create folders
foreach ($folder in $folders) {
    if (-Not (Test-Path $folder)) {
        New-Item -ItemType Directory -Path $folder -Force | Out-Null
        Write-Host "Created: $folder"
    } else {
        Write-Host "Already exists: $folder"
    }
}

# Create base files
$files = @(
    "$root\ocr\recognizer.py",
    "$root\caption\minigpt4_cpu.py",
    "$root\retriever\embedder.py",
    "$root\retriever\vector_db.py",
    "$root\chatbot\graph_config.py",
    "$root\chatbot\nodes\ocr_node.py",
    "$root\chatbot\nodes\caption_node.py",
    "$root\chatbot\nodes\merge_node.py",
    "$root\chatbot\nodes\embed_node.py",
    "$root\chatbot\nodes\query_node.py",
    "$root\chatbot\nodes\rag_node.py",
    "$root\chatbot\nodes\answer_node.py",
    "$root\app\main.py",
    "$root\frontend\index.html",
    "$root\frontend\app.js",
    "$root\frontend\styles.css",
    "$root\utils\helpers.py",
    "$root\docker\Dockerfile",
    "$root\notebooks\playground.ipynb",
    "$root\requirements.txt",
    "$root\.env",
    "$root\README.md"
)

# Create empty files
foreach ($file in $files) {
    if (-Not (Test-Path $file)) {
        New-Item -ItemType File -Path $file -Force | Out-Null
        Write-Host "Created file: $file"
    } else {
        Write-Host "File exists: $file"
    }
}

Write-Host "`n Folder structure initialized successfully at: $root"
