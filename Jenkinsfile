pipeline {
  agent any
  options { skipDefaultCheckout(true); timestamps() }
  stages {
    stage('Checkout') { steps { checkout scm } }
    stage('Install') { steps { sh 'python -m pip install --upgrade pip'; sh 'python -m pip install -r requirements.txt' } }
    stage('Tests') { steps { sh 'pytest -v tests/' } }
    stage('Short Run') { steps { sh 'python main.py' } }
  }
}
