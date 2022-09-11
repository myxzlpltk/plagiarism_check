# Plagiarism Check Project using Vector Based System
In this project, we will use several tech stack. such as: `Flask`, `MilvusDB`

## How to run
### 1. Clone this repo
```bash
git clone https://github.com/myxzlpltk/plagiarism_check.git
```

### 2. Install all dependencies
Enter project folder
```bash
cd plagiarism_check
```
Create conda virtual environment
```bash
conda env create -f environment.yml
```
Activate environment and setup for your own IDE
```bash
conda activate nlp
```

### 3. Install MilvusDB (Docker)
Make sure you have docker desktop installed. If not, please install it first.
```bash
docker-compose up -d
```
You can also install [Attu](https://github.com/zilliztech/attu/releases) for managing database.
Reference: https://milvus.io/docs/v2.1.x/install_standalone-docker.md

### 4. Train the model
Open `Plagiarism.ipynb` and run all cells.

### 5. Run the server
```bash
python -m flask --app server --debug run
```

## How to use
1. Upload the file
2. Get the plagiarism report