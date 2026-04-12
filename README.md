# BePro AI/ML Course

## Getting Started (for students)

### 1. Install Git
- **Windows**: Download from [git-scm.com](https://git-scm.com/download/win)
- **Mac**: Run `xcode-select --install` in Terminal

### 2. Clone this repo
```bash
git clone https://github.com/bepro-aiml/burgut.git
cd burgut
```

### 3. Submit an assignment
```bash
# Create your branch (use your name)
git checkout -b firstname-lastname

# Go to the assignment folder
cd module-1/class-1/submissions

# Create your folder
mkdir firstname-lastname
cd firstname-lastname

# Add your files here, then:
git add .
git commit -m "Module 1 Class 1 submission - Firstname Lastname"
git push origin firstname-lastname
```

### 4. Open a Pull Request
1. Go to the repo on GitHub
2. Click "Compare & pull request"
3. Title: `Module X Class Y - Firstname Lastname`
4. Click "Create pull request"

### Rules
- Do NOT push directly to `main`
- Always create your own branch
- Put your work inside `submissions/firstname-lastname/`
- One PR per assignment
