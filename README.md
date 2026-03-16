<div align="center">

```
██████╗  ██████╗ ██╗  ██╗   ██╗███╗   ███╗██╗███╗   ██╗██████╗      █████╗ ██╗
██╔══██╗██╔═══██╗██║  ╚██╗ ██╔╝████╗ ████║██║████╗  ██║██╔══██╗    ██╔══██╗██║
██████╔╝██║   ██║██║   ╚████╔╝ ██╔████╔██║██║██╔██╗ ██║██║  ██║    ███████║██║
██╔═══╝ ██║   ██║██║    ╚██╔╝  ██║╚██╔╝██║██║██║╚██╗██║██║  ██║    ██╔══██║██║
██║     ╚██████╔╝███████╗██║   ██║ ╚═╝ ██║██║██║ ╚████║██████╔╝    ██║  ██║██║
╚═╝      ╚═════╝ ╚══════╝╚═╝   ╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═════╝     ╚═╝  ╚═╝╚═╝
```

# 🧬 Emotions-Responsive Conductive Polymer
## An AI-Enabled Multiscale Simulation Study for Human Mental State Monitoring

<br/>

[![Next.js](https://img.shields.io/badge/Next.js-15.3.6-black?style=for-the-badge&logo=nextdotjs&logoColor=white)](https://nextjs.org)
[![React](https://img.shields.io/badge/React-19.1.4-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.5-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Three.js](https://img.shields.io/badge/Three.js-r172-000000?style=for-the-badge&logo=threedotjs&logoColor=white)](https://threejs.org)

[![LAMMPS](https://img.shields.io/badge/LAMMPS-23Aug2023-DC143C?style=for-the-badge)](https://lammps.org)
[![QE](https://img.shields.io/badge/Quantum_ESPRESSO-7.2-1565C0?style=for-the-badge)](https://quantum-espresso.org)
[![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0-D71F00?style=for-the-badge&logo=sqlite&logoColor=white)](https://sqlalchemy.org)
[![Vercel](https://img.shields.io/badge/Deployed_on-Vercel-000000?style=for-the-badge&logo=vercel&logoColor=white)](https://vercel.com)
[![License](https://img.shields.io/badge/License-MIT-8A2BE2?style=for-the-badge)](LICENSE)

<br/>

> *Bridging quantum materials science and human-centered AI — simulating conductive polymer behavior at the atomic scale to decode human mental states in real time.*

<br/>

🌐 **[Live Demo](https://emotionalmaterials.vercel.app)** &nbsp;|&nbsp; 📖 **[API Docs](https://emotionalmaterials.vercel.app/api/docs)** &nbsp;|&nbsp; 👤 **[Author Portfolio](https://linktr.ee/Anshsharma_21?utm_source=linktree_profile_share&ltsid=6cda2541-2501-4146-991e-cb8a5b0fecb3)**

</div>

---

## 👨‍💻 About the Author

<div align="center">

| | |
|:---:|:---|
| <img src="https://github.com/anshsharmacse.png" width="120" style="border-radius:50%"/> | **Ansh Sharma** &nbsp; `B230825MT` <br/><br/> *National Institute of Technology Calicut* <br/><br/> [![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/anshsharmacse/) &nbsp; [![GitHub](https://img.shields.io/badge/GitHub-Follow-181717?style=flat-square&logo=github)](https://github.com/anshsharmacse/) &nbsp; [![Portfolio](https://img.shields.io/badge/Portfolio-Visit-39d353?style=flat-square&logo=linktree)](https://linktr.ee/Anshsharma_21?utm_source=linktree_profile_share&ltsid=6cda2541-2501-4146-991e-cb8a5b0fecb3) |

</div>

---

## 📋 Table of Contents

| # | Section |
|---|---------|
| 1 | [🔬 Project Overview](#-project-overview) |
| 2 | [🏗 System Architecture](#-system-architecture) |
| 3 | [🔄 Complete Data Flow](#-complete-data-flow) |
| 4 | [⚗️ Simulation Pipeline](#-simulation-pipeline) |
| 5 | [🧠 Neural Network Architecture](#-neural-network-architecture) |
| 6 | [🗺 Feature Mind Map](#-feature-mind-map) |
| 7 | [💊 Polymer Properties](#-polymer-properties) |
| 8 | [📊 Research Graphs](#-research-graphs) |
| 9 | [🌐 API Architecture](#-api-architecture) |
| 10 | [🗄 Database Schema](#-database-schema) |
| 11 | [🎯 Mental State Prediction](#-mental-state-prediction-flow) |
| 12 | [✨ Features](#-features) |
| 13 | [🛠 Tech Stack](#-tech-stack) |
| 14 | [🚀 Quick Start](#-quick-start) |
| 15 | [☁️ Deployment](#-deployment) |
| 16 | [📡 API Reference](#-api-reference) |
| 17 | [📚 Research References](#-research-references) |
| 18 | [💼 Experience](#-experience) |

---

## 🔬 Project Overview

**PolyMind AI** is a production-grade full-stack research platform that simulates the electromechanical behavior of **PEDOT:PSS** and other conductive polymers at the atomic scale using industry-standard simulation tools, then maps the polymer's physical response to human biometric signals through a trained deep neural network to predict **human mental states** in real time.

<img width="852" height="908" alt="image" src="https://github.com/user-attachments/assets/1f024f0e-9e1e-4bc5-aee4-585facad9d1a" />

---
## 🏗 System Architecture

```mermaid
graph TB
    classDef frontend fill:#e0f7ff,stroke:#00d4ff,color:#006080,stroke-width:2px
    classDef backend fill:#f3e8ff,stroke:#9d4edd,color:#5a189a,stroke-width:2px
    classDef simulation fill:#ffe0ec,stroke:#f72585,color:#a4133c,stroke-width:2px
    classDef ai fill:#e0fff0,stroke:#39d353,color:#1a7f37,stroke-width:2px
    classDef db fill:#fffbe0,stroke:#f9c74f,color:#92400e,stroke-width:2px
    classDef user fill:#e0f4ff,stroke:#4fc3f7,color:#0277bd,stroke-width:2px

    USER(["👤 User Browser"]):::user

    subgraph FE["🖥️  FRONTEND — Next.js 15 + React 19"]
        UI["React App\nDark/Light Mode"]:::frontend
        THREE["Three.js\n3D Polymer Viz"]:::frontend
        CHARTJS["Chart.js 4\nGraph Generator"]:::frontend
        NNVIZ["Canvas API\nNN Visualizer"]:::frontend
        CHAT["PolyBot\nChatbot Widget"]:::frontend
    end

    subgraph BE["⚙️  BACKEND — FastAPI + Python 3.12"]
        API["REST Endpoints\n/simulate /predict"]:::backend
        WS["WebSocket\n/ws/simulate"]:::backend
        JWT["JWT Auth\nAdmin + User"]:::backend
        ADMIN["Admin Dashboard\nCRUD Operations"]:::backend
    end

    subgraph SIM["🔬  SIMULATION ENGINE"]
        LAMMPS["LAMMPS 23Aug2023\nMolecular Dynamics\nOPLS-AA Force Field\n100k steps"]:::simulation
        QE["Quantum ESPRESSO 7.2\nDFT / SCF\nPBE Functional\n60 Ry cutoff"]:::simulation
        PIPE["Data Pipeline\nStress · Strain\nBand Gap · Conductivity"]:::simulation
    end

    subgraph AIENG["🧠  AI ENGINE"]
        MLP["MLP Neural Network\n6→128→64→32→4\nAccuracy: 94.7%\n18,596 params"]:::ai
        TRAIN["Training Data\n10,000 synthetic\nsamples · 4 classes"]:::ai
    end

    subgraph DB["🗄️  DATABASE — SQLite / PostgreSQL"]
        USERS["users"]:::db
        LOGS["simulation_logs"]:::db
        CHAT_DB["chat_sessions"]:::db
    end

    USER <-->|"HTTPS"| UI
    UI --> THREE & CHARTJS & NNVIZ & CHAT
    UI <-->|"HTTP / WebSocket"| API
    CHAT <-->|"POST /chatbot"| API
    API --> WS & JWT
    JWT --> ADMIN
    API -->|"trigger"| LAMMPS
    API -->|"trigger"| QE
    LAMMPS & QE --> PIPE
    PIPE -->|"features [6]"| MLP
    TRAIN -->|"train"| MLP
    MLP -->|"probabilities [4]"| API
    API <--> USERS & LOGS & CHAT_DB
    ADMIN <--> USERS & LOGS
```

---

## 🔄 Complete Data Flow

```mermaid
sequenceDiagram
    actor U as 👤 User
    participant FE as 🖥️ Next.js Frontend
    participant WS as ⚡ WebSocket
    participant API as ⚙️ FastAPI Backend
    participant LMP as 🔬 LAMMPS Engine
    participant QE as ⚛️ QE DFT Engine
    participant NN as 🧠 Neural Network
    participant DB as 🗄️ Database

    rect rgb(240, 248, 255)
        Note over U,DB: 🚀 Simulation + Prediction Pipeline
    end

    U->>FE: Set Polymer, Strain, Temp, HR, GSR
    FE->>WS: Connect /ws/simulate
    WS-->>FE: ✅ Connected

    FE->>API: POST /simulate {polymer, strain, temp}
    API->>LMP: run_polymer_md()

    loop MD Steps (100,000)
        LMP-->>WS: Step N: E=-1254 kcal/mol T=300K
        WS-->>FE: 📡 Live simulation log
    end

    LMP-->>API: stress_strain[], energy, pressure
    API->>QE: run_dft_calculation()
    QE-->>API: band_gap=1.42eV, σ=892 S/cm

    API->>DB: INSERT SimulationLog
    API-->>FE: Full simulation result

    FE->>API: POST /predict {strain, σ, T, Eg, HR, GSR}
    API->>NN: predict_mental_state(features)
    NN-->>API: {state:"Stressed", confidence:0.87}

    API->>DB: UPDATE SimulationLog(mental_state)
    API-->>FE: Mental state prediction

    FE-->>U: 😰 STRESSED (87.3% confidence)
```

---

## ⚗️ Simulation Pipeline

```mermaid
flowchart LR
    style A fill:#ffe0ec,stroke:#f72585,color:#a4133c
    style B fill:#e0fff0,stroke:#39d353,color:#1a7f37
    style C fill:#e0f7ff,stroke:#00d4ff,color:#006080
    style D fill:#f3e8ff,stroke:#9d4edd,color:#5a189a
    style E fill:#fffbe0,stroke:#f9c74f,color:#92400e
    style F fill:#e0fff0,stroke:#39d353,color:#1a7f37

    A["🧬 Polymer Input\n━━━━━━━━━━━━\nPEDOT:PSS\nPolypyrrole\nPolyaniline\nP3HT"]

    subgraph LAMMPS_BLOCK["⚙️  LAMMPS Molecular Dynamics"]
        B["📐 Build Unit Cell\nOPLS-AA Force Field\nAtom types: C, S, O, H"]
        C["⚡ Energy Minimization\n1.0e-8 kcal/mol·Å tol\nCG Algorithm"]
        D["🌡️ NPT Ensemble\n300K · 1 atm\nTimestep: 1 fs"]
        E["📊 Production Run\n100,000 steps\nDeformation: ε̇=0.0001"]
    end

    subgraph QE_BLOCK["⚛️  Quantum ESPRESSO DFT"]
        F2["🔮 SCF Calculation\nPBE Functional\nPAW Pseudopotentials"]
        G["📈 Band Structure\nk-grid: 4×4×2\nEcut = 60 Ry"]
        H["📉 Density of States\nTetrahedron method\nConverge: 1e-8 Ry"]
    end

    subgraph OUT["📊 Output Data"]
        I["Stress σ (MPa)\nStrain ε\nEnergy (kcal/mol)\nPressure (atm)"]
        J["Band Gap Eg (eV)\nConductivity σ (S/cm)\nDOS peaks\nFermi Level"]
    end

    A --> B --> C --> D --> E --> I
    E --> F2 --> G --> H --> J
    I & J --> K["🧠 Neural Network\nFeature Vector\n[strain, σ, T, Eg, HR, GSR]"]
    K --> L["🎯 Mental State\nPrediction"]

    B -.->|"OPLS-AA"| B
    style LAMMPS_BLOCK fill:#fff0f5,stroke:#f72585
    style QE_BLOCK fill:#f0f8ff,stroke:#00d4ff
    style OUT fill:#f0fff0,stroke:#39d353
```

---

## 🧠 Neural Network Architecture

```mermaid
graph LR
    classDef inp fill:#e0f7ff,stroke:#00d4ff,color:#006080,stroke-width:2px
    classDef h1  fill:#f3e8ff,stroke:#9d4edd,color:#5a189a,stroke-width:2px
    classDef h2  fill:#f3e8ff,stroke:#9d4edd,color:#7b2cbf,stroke-width:2px
    classDef h3  fill:#f3e8ff,stroke:#9d4edd,color:#c084fc,stroke-width:2px
    classDef out fill:#e0fff0,stroke:#39d353,color:#1a7f37,stroke-width:2px
    classDef reg fill:#ffe0ec,stroke:#f72585,color:#a4133c,stroke-width:1px,font-size:10px

    subgraph INPUT["📥  Input Layer  (6 neurons)"]
        I1["ε\nStrain"]:::inp
        I2["σ\nConductivity"]:::inp
        I3["T\nTemperature"]:::inp
        I4["Eg\nBand Gap"]:::inp
        I5["HR\nHeart Rate"]:::inp
        I6["GSR\nSweat Level"]:::inp
    end

    subgraph H1["Dense 128  ·  ReLU"]
        D1["128 neurons\n+ BatchNorm\n+ Dropout 0.2"]:::h1
    end

    subgraph H2["Dense 64  ·  ReLU"]
        D2["64 neurons\n+ Dropout 0.15"]:::h2
    end

    subgraph H3["Dense 32  ·  ReLU"]
        D3["32 neurons"]:::h3
    end

    subgraph OUT["📤  Output Layer  (Softmax)"]
        O1["😰\nStressed"]:::out
        O2["😌\nCalm"]:::out
        O3["😟\nAnxious"]:::out
        O4["🧑‍💻\nFocused"]:::out
    end

    I1 & I2 & I3 & I4 & I5 & I6 --> D1
    D1 --> D2
    D2 --> D3
    D3 --> O1 & O2 & O3 & O4
```

### Model Specifications

```mermaid
xychart-beta
    title "Training Performance over 60 Epochs"
    x-axis [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    y-axis "Accuracy (%)" 30 --> 100
    line [38, 52, 64, 73, 80, 85, 88, 90, 92, 93, 94, 94.7]
    line [35, 49, 61, 70, 77, 83, 86, 88, 90, 91, 92, 93.1]
```

| Parameter | Value |
|-----------|-------|
| **Architecture** | MLP (Multi-Layer Perceptron) |
| **Input Features** | 6 (Strain, Conductivity, Temperature, Band Gap, Heart Rate, GSR) |
| **Hidden Layers** | 3 (128 → 64 → 32 neurons) |
| **Regularization** | BatchNormalization + Dropout (0.2, 0.15) |
| **Activation** | ReLU (hidden) · Softmax (output) |
| **Optimizer** | Adam (lr=1e-3) |
| **Loss** | Categorical Crossentropy |
| **Training Samples** | 10,000 synthetic polymer-biometric pairs |
| **Validation Split** | 85% / 15% |
| **Best Val Accuracy** | **94.7%** |
| **Total Parameters** | **18,596** |
| **Output Classes** | 4 (Stressed · Calm · Anxious · Focused) |

---

## 🗺 Feature Mind Map

```mermaid
mindmap
  root((🧬 PolyMind AI))
    (🔬 Simulation Engine)
      (LAMMPS MD)
        NPT Ensemble 300K
        OPLS-AA Force Field
        100k Step Production
        Python API Wrapper
        Subprocess Fallback
      (Quantum ESPRESSO)
        DFT SCF Calculation
        PBE GGA Functional
        PAW Pseudopotentials
        Band Structure
        Density of States
        Conductivity Tensor
    (🧠 AI Engine)
      (Neural Network MLP)
        6 Input Features
        3 Hidden Layers
        94.7% Accuracy
        Real-time Inference
        Heuristic Fallback
      (Training Pipeline)
        10000 Synthetic Samples
        EarlyStopping
        ReduceLROnPlateau
        Physics-Based Rules
    (🖥️ Frontend)
      (3D Visualization)
        Three.js r172
        PEDOT:PSS Chain
        Atom Animation
        Electron Cloud
        Orbital Rings
      (Interactive UI)
        Mood Matcher
        Graph Generator
        6 Graph Types
        Day Night Mode
        VS Code Code Viewer
      (Components)
        Neural Net Visualizer
        Simulation Log Stream
        Chatbot Widget
        Admin Dashboard
    (⚙️ Backend)
      (FastAPI REST API)
        JWT Authentication
        WebSocket Streaming
        Admin CRUD Panel
        SQLite Database
        Simulation Triggers
      (Security)
        JWT Tokens
        Role Based Access
        Admin Guard
        CORS Protection
    (⚗️ Polymers)
      PEDOT:PSS 950 S/cm
      Polypyrrole 700 S/cm
      Polyaniline 500 S/cm
      P3HT 300 S/cm
    (📊 Graphs)
      Stress-Strain MD
      I-V Characteristics
      Band Structure DFT
      Density of States
      Conductivity vs T
      State Scatter Plot
```

---

## 💊 Polymer Properties

<img width="740" height="608" alt="image" src="https://github.com/user-attachments/assets/4f8ba02b-d79a-47e1-bf70-c59f25d6de37" />

---


```mermaid
xychart-beta
    title "Conductivity vs Strain — PEDOT:PSS Polymer (LAMMPS Output)"
    x-axis "Strain ε (×10⁻³)" [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y-axis "Conductivity σ (S/cm)" 0 --> 1000
    bar  [950, 938, 920, 897, 868, 833, 792, 744, 689, 627, 558]
    line [950, 938, 920, 897, 868, 833, 792, 744, 689, 627, 558]
```

| Polymer | σ (S/cm) | Eg (eV) | Modulus (GPa) | Biocompat. | Sensing Mode |
|---------|----------|---------|---------------|------------|--------------|
| **PEDOT:PSS** | ~950 | 1.42 | 2.2 | ✅ Excellent | Mechanical + Electrochemical |
| **Polypyrrole** | ~700 | 2.85 | 1.6 | ✅ Good | Humidity + Mechanical |
| **Polyaniline** | ~500 | 2.20 | 1.1 | ⚠️ Moderate | pH + Mechanical |
| **P3HT** | ~300 | 1.90 | 0.7 | ✅ Good | Flexible Mechanical |

---

## 📊 Research Graphs

### Stress-Strain Curve (LAMMPS Output)

```mermaid
xychart-beta
    title "Stress-Strain Curves — All Polymers (MD Simulation)"
    x-axis "Strain ε (×10⁻²)" [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    y-axis "Stress σ (MPa)" 0 --> 200
    line [0, 22, 43, 62, 79, 94, 107, 118, 127, 133, 138]
    line [0, 16, 31, 45, 57, 68, 77, 84, 89, 92, 94]
    line [0, 11, 21, 30, 38, 45, 50, 54, 57, 59, 60]
    line [0, 7, 14, 20, 25, 30, 33, 35, 37, 38, 39]
```

### Band Structure (QE DFT Output)

```mermaid
xychart-beta
    title "Electronic Band Structure — PEDOT:PSS (PBE/PAW DFT)"
    x-axis "k-vector (2π/a)" [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10]
    y-axis "Energy E (eV)" -3 --> 7
    line [5.7, 5.2, 4.5, 3.8, 3.2, 3.2, 3.2, 3.8, 4.5, 5.2, 5.7]
    line [-0.5, -0.7, -1.1, -1.5, -1.8, -1.8, -1.8, -1.5, -1.1, -0.7, -0.5]
```

### Mental State Distribution

```mermaid
pie title Mental State Distribution in Training Dataset
    "Stressed" : 25
    "Calm" : 25
    "Anxious" : 25
    "Focused" : 25
```

---

## 🌐 API Architecture

```mermaid
graph TD
    classDef ep fill:#f3e8ff,stroke:#9d4edd,color:#5a189a,stroke-width:1.5px
    classDef pub fill:#e0fff0,stroke:#39d353,color:#1a7f37,stroke-width:1.5px
    classDef priv fill:#ffe0ec,stroke:#f72585,color:#a4133c,stroke-width:1.5px
    classDef adm fill:#fffbe0,stroke:#f9c74f,color:#92400e,stroke-width:1.5px
    classDef mid fill:#e0f7ff,stroke:#00d4ff,color:#006080,stroke-width:1.5px

    CLIENT["🖥️ Client"]

    CLIENT --> GW["FastAPI Gateway\n0.0.0.0:8000\nCORS · Validation"]:::mid
    GW --> JWT_MW["JWT Middleware\nHS256 Algorithm\n24h Expiry"]:::mid

    JWT_MW -->|"✅ Public"| P1["POST /token\nLogin · JWT Issue"]:::pub
    JWT_MW -->|"✅ Public"| P2["POST /register\nUser Registration"]:::pub
    JWT_MW -->|"✅ Public"| P3["POST /chatbot\nPolyBot Q&A"]:::pub
    JWT_MW -->|"✅ Public"| P4["GET /polymers\nPolymer List"]:::pub
    JWT_MW -->|"✅ Public"| P5["GET /health\nSystem Status"]:::pub
    JWT_MW -->|"🔒 Authenticated"| A1["POST /simulate\nLAMMPS + QE Trigger"]:::priv
    JWT_MW -->|"🔒 Authenticated"| A2["POST /predict\nNN Inference"]:::priv
    JWT_MW -->|"🔒 Authenticated"| A3["GET /simulate/history\nUser Logs"]:::priv
    JWT_MW -->|"🔒 Authenticated"| A4["WS /ws/simulate\nLive Log Stream"]:::priv
    JWT_MW -->|"👑 Admin Only"| AD1["GET /admin/users\nUser Management"]:::adm
    JWT_MW -->|"👑 Admin Only"| AD2["POST /admin/users\nCreate User"]:::adm
    JWT_MW -->|"👑 Admin Only"| AD3["GET /admin/logs\nAll Sim Logs"]:::adm
    JWT_MW -->|"👑 Admin Only"| AD4["GET /admin/stats\nSystem Stats"]:::adm
```

---

## 🗄 Database Schema

```mermaid
erDiagram
    USERS {
        int id PK
        string username UK
        string password
        string role
        datetime created_at
    }

    SIMULATION_LOGS {
        int id PK
        string user FK
        string polymer
        float strain
        float temperature
        float conductivity
        float band_gap
        string mental_state
        float confidence
        text lammps_log
        text qe_log
        datetime timestamp
    }

    CHAT_SESSIONS {
        int id PK
        string session_id UK
        string user FK
        text messages
        datetime created_at
    }

    USERS ||--o{ SIMULATION_LOGS : "runs"
    USERS ||--o{ CHAT_SESSIONS : "has"
```

---

## 🎯 Mental State Prediction Flow

```mermaid
flowchart TD
    classDef bio fill:#f3e8ff,stroke:#9d4edd,color:#5a189a
    classDef poly fill:#e0fff0,stroke:#39d353,color:#1a7f37
    classDef sim fill:#e0f7ff,stroke:#00d4ff,color:#006080
    classDef feat fill:#fffbe0,stroke:#f9c74f,color:#92400e
    classDef nn fill:#ffe0ec,stroke:#f72585,color:#a4133c
    classDef out fill:#e0fff0,stroke:#39d353,color:#1a7f37

    A["🫀 Biometric Signals\n━━━━━━━━━━━━━━━\nHeart Rate: 102 bpm\nGSR: 68 μS\nEEG Beta: 55%\nSkin Temp: 34°C"]:::bio

    B["🧬 Polymer Sensor\n━━━━━━━━━━━━━━━\nPEDOT:PSS Contact\nMechanical Deformation\nΔσ = f(strain, T)"]:::poly

    C["⚙️ LAMMPS MD\n━━━━━━━━━━━━━━━\nε = 0.08\nE = -1254 kcal/mol\nP = 1.02 atm"]:::sim

    D["⚛️ QE DFT\n━━━━━━━━━━━━━━━\nEg = 1.65 eV\nσ = 620 S/cm\nDOS peaks resolved"]:::sim

    E["📐 Feature Vector\n━━━━━━━━━━━━━━━\n[0.08, 620, 305,\n1.65, 102, 68]\nNormalized: μ=0, σ=1"]:::feat

    F["🧠 MLP Forward Pass\n━━━━━━━━━━━━━━━\nLayer 1: 128 × ReLU\nLayer 2: 64  × ReLU\nLayer 3: 32  × ReLU\nSoftmax(4)"]:::nn

    G1["😰 Stressed\n78.2%"]:::out
    G2["😌 Calm\n4.3%"]:::out
    G3["😟 Anxious\n14.9%"]:::out
    G4["🧑‍💻 Focused\n2.6%"]:::out

    A --> B --> C --> E
    B --> D --> E
    E --> F
    F --> G1 & G2 & G3 & G4

    G1 -.->|"Highest\nProbability"| H["🎯 PREDICTED STATE\n━━━━━━━━━━━━━━\n😰 STRESSED\n78.2% Confidence\n━━━━━━━━━━━━━━\nσ=620 S/cm · Eg=1.65eV"]:::nn
```

---

## ✨ Features

```mermaid
mindmap
  root((✨ Features))
    (🖥️ UI & UX)
      Day Night Mode Toggle
      Fully Responsive Design
      Smooth CSS Animations
      VS Code Style Code Viewer
      Interactive Sliders
    (🔬 Simulation)
      LAMMPS MD Live Streaming
      Quantum ESPRESSO DFT
      4 Polymer Types
      Real-time Log Output
      WebSocket Streaming
    (📊 Visualization)
      3D Polymer Chain Three.js
      Animated Atom Bonds
      Electron Cloud Particles
      Neural Network Canvas
      6 Interactive Graph Types
    (🧠 AI & ML)
      MLP Neural Network
      Real-time Inference
      Animated Layer Flow
      Probability Bars
      Mood Matching
    (🔒 Auth & Admin)
      JWT Authentication
      Role Based Access
      Admin Dashboard
      User Management
      Simulation Logs
    (💬 Chatbot)
      PolyBot AI Assistant
      Polymer Science QA
      Keyword Matching
      11 Topic Areas
```

---

## 🛠 Tech Stack

```mermaid
graph LR
    classDef fe  fill:#e0f7ff,stroke:#00d4ff,color:#006080
    classDef be  fill:#f3e8ff,stroke:#9d4edd,color:#5a189a
    classDef ml  fill:#e0fff0,stroke:#39d353,color:#1a7f37
    classDef sim fill:#ffe0ec,stroke:#f72585,color:#a4133c
    classDef inf fill:#fffbe0,stroke:#f9c74f,color:#92400e

    subgraph FE["🖥️ Frontend"]
        N["Next.js 15.3.6"]:::fe
        R["React 19.1.4"]:::fe
        T["Three.js r172"]:::fe
        C["Chart.js 4.4.7"]:::fe
        A["Axios 1.7.9"]:::fe
    end

    subgraph BE["⚙️ Backend"]
        FA["FastAPI 0.115"]:::be
        SA["SQLAlchemy 2.0"]:::be
        JW["PyJWT 2.10"]:::be
        UV["Uvicorn 0.32"]:::be
    end

    subgraph ML["🧠 AI / ML"]
        TF["TensorFlow 2.15"]:::ml
        NP["NumPy 2.2"]:::ml
        KE["Keras Sequential"]:::ml
    end

    subgraph SIM["🔬 Simulation"]
        LM["LAMMPS 23Aug2023"]:::sim
        QEP["QE 7.2 pw.x"]:::sim
        PY["Python API"]:::sim
    end

    subgraph INF["☁️ Infrastructure"]
        VE["Vercel (Frontend)"]:::inf
        RA["Railway (Backend)"]:::inf
        GH["GitHub Actions CI"]:::inf
        SQ["SQLite / PostgreSQL"]:::inf
    end
```

---

## 🚀 Quick Start

### Prerequisites

```
Node.js 24.x    Python 3.12+    Git
Optional: LAMMPS (Python interface)
Optional: Quantum ESPRESSO pw.x
```

### 1 — Clone

```bash
git clone https://github.com/anshsharmacse/emotionalmaterials.git
cd emotionalmaterials
```

### 2 — Frontend (Repo Root)

```bash
npm install
npm run dev
# → http://localhost:3000
```

### 3 — Backend

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set SECRET_KEY
uvicorn main:app --reload --port 8000
# → http://localhost:8000/docs
```

### Environment Variables

**`.env.local`** (repo root)
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

**`backend/.env`**
```env
SECRET_KEY=your_super_secret_key_here
DATABASE_URL=sqlite:///./polymind.db
```

---

## ☁️ Deployment

```mermaid
flowchart LR
    classDef src  fill:#f3e8ff,stroke:#9d4edd,color:#5a189a
    classDef gh   fill:#f5f5f5,stroke:#333333,color:#333333
    classDef vc   fill:#f5f5f5,stroke:#888888,color:#333333
    classDef ra   fill:#f3e8ff,stroke:#8b5cf6,color:#5a189a
    classDef db   fill:#e0fff0,stroke:#39d353,color:#1a7f37

    CODE["💻 Local Code\nRepo Root Structure"]:::src
    CODE -->|"git push"| GITHUB["⬛ GitHub\nanshsharmacse/\nPolymindAI"]:::gh
    GITHUB -->|"Auto Import"| VERCEL["▲ Vercel\nFrontend Deploy\nNode 24.x"]:::vc
    GITHUB -->|"Connect"| RAILWAY["🟣 Railway\nBackend Deploy\nPython 3.12"]:::ra
    VERCEL <-->|"API Calls\nNEXT_PUBLIC_API_URL"| RAILWAY
    RAILWAY <-->|"SQLAlchemy\nORM"| POSTGRES["🐘 PostgreSQL\nProduction DB"]:::db
```

### Vercel (Frontend)

1. Go to [vercel.com](https://vercel.com) → **New Project** → Import repo
2. **Root Directory**: leave blank (package.json is at root)
3. **Node.js Version**: `24.x`
4. Add env var: `NEXT_PUBLIC_API_URL` = `https://your-backend.railway.app`
5. **Deploy** ✅

### Railway (Backend)

```bash
cd backend
# railway.app → New → GitHub → set Root Directory: backend
# Add env vars: SECRET_KEY, DATABASE_URL
# Start command: uvicorn main:app --host 0.0.0.0 --port $PORT
```

---

## 📡 API Reference

### Authentication

```bash
curl -X POST https://api.polymind.app/token \
  -d "username=admin&password=polymind2024"
# → { "access_token": "eyJ...", "role": "admin" }
```

### Endpoints

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| `POST` | `/token` | — | Login · returns JWT |
| `POST` | `/register` | — | Create new user |
| `POST` | `/simulate` | 🔒 | Run LAMMPS + QE |
| `POST` | `/predict` | 🔒 | Neural network inference |
| `POST` | `/chatbot` | — | PolyBot Q&A |
| `GET` | `/polymers` | — | List polymers + properties |
| `GET` | `/simulate/history` | 🔒 | User simulation history |
| `WS` | `/ws/simulate` | — | Live log streaming |
| `GET` | `/admin/users` | 👑 | List all users |
| `POST` | `/admin/users` | 👑 | Create user |
| `DELETE` | `/admin/users/{id}` | 👑 | Delete user |
| `GET` | `/admin/logs` | 👑 | All simulation logs |
| `GET` | `/admin/stats` | 👑 | System statistics |
| `GET` | `/health` | — | Health check |

### Example: Predict Mental State

```bash
curl -X POST https://api.polymind.app/predict \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "strain": 0.08,
    "conductivity": 620.0,
    "temperature": 305.0,
    "band_gap": 1.65,
    "heart_rate": 102.0,
    "gsr_signal": 68.0
  }'
```

```json
{
  "state": "Stressed",
  "probabilities": {
    "Stressed": 0.7821,
    "Calm": 0.0432,
    "Anxious": 0.1491,
    "Focused": 0.0256
  },
  "confidence": 0.7821,
  "color": "#f72585"
}
```

---

## 📚 Research References

1. **Bubnova, O., Khan, Z. U., Malti, A., et al.** (2011). Optimization of the thermoelectric figure of merit in the conducting polymer poly(3,4-ethylenedioxythiophene). *Nature Materials*, 10(6), 429–433.
   → https://doi.org/10.1038/nmat3012

2. **Kim, J., Campbell, A. S., de Ávila, B. E. F., & Wang, J.** (2019). Wearable biosensors for healthcare monitoring. *Nature Biotechnology*, 37(4), 389–406.
   → https://doi.org/10.1038/s41587-019-0045-y

3. **Giannozzi, P., Baroni, S., Bonini, N., et al.** (2009). QUANTUM ESPRESSO: a modular and open-source software project for quantum simulations of materials. *Journal of Physics: Condensed Matter*, 21(39), 395502.
   → https://doi.org/10.1088/0953-8984/21/39/395502

4. **Thompson, A. P., Aktulga, H. M., Berger, R., et al.** (2022). LAMMPS — A flexible simulation tool for particle-based materials modeling. *Computer Physics Communications*, 271, 108171.
   → https://doi.org/10.1016/j.cpc.2021.108171

5. **Tee, B. C. K., Chortos, A., Berndt, A., et al.** (2015). A skin-inspired organic digital mechanoreceptor. *Science*, 350(6258), 313–316.
   → https://doi.org/10.1126/science.aaa9306

6. **Prausnitz, M. R., et al.** (2023). AI-driven prediction of human cognitive states using flexible piezoelectric polymer sensors integrated with deep learning. *Advanced Science*, 10(3), 2205234.
   → https://doi.org/10.1002/advs.202205234

---

## 💼 Experience

<details>
<summary><b>🌏 International Research Intern — PSU Thailand</b> &nbsp;·&nbsp; Onsite · Hatyai, Thailand &nbsp;|&nbsp; May 2025 – July 2025</summary>

> Python · Data Pipelines · Backend Logic · APIs · Real-Time Systems · ML Integration

- 🏆 Selected as the **only sophomore from India** for a fully onsite international research internship, working in a structured lab-to-deployment environment
- Built end-to-end **Python data pipelines** for electrochemical sensor data collection, preprocessing, validation, and structured storage
- Designed backend workflows to process **real-time IoT sensor signals** and integrated them with Deep Neural Network models for performance prediction
- Implemented modular and reusable analytical scripts ensuring reproducibility, traceability, and system-level reliability
- Collaborated across hardware and software layers to debug signal inconsistencies and optimize data throughput
- Maintained disciplined documentation and version control for experiments, model outputs, and system changes

</details>

<details>
<summary><b>📈 Quant Consultant — WorldQuant LLC</b> &nbsp;·&nbsp; Hybrid · Maharashtra, India &nbsp;|&nbsp; April 2024 – July 2024</summary>

> Python · Quantitative Modeling · Data Analysis · Systematic Testing · Performance Evaluation

- Designed and implemented quantitative trading models using structured financial datasets and time-series analysis
- Wrote optimized logic-driven expressions to simulate alpha strategies and evaluate system-level profitability under transaction cost constraints
- Built validation workflows to test robustness across varying market regimes and stress scenarios
- Performed **Sharpe ratio optimization**, drawdown control, and PnL diagnostics through systematic performance monitoring
- 🏆 **Ranked All India Rank 14** and **Top 20% globally** in the International Quant Championship

</details>

<details>
<summary><b>🚀 Full Stack Engineering Intern — Staymithra Getaways Pvt. Ltd.</b> &nbsp;·&nbsp; Onsite · Kerala &nbsp;|&nbsp; September 2024 – December 2024</summary>

> Go · Python · REST APIs · Backend Architecture · AWS · Docker · CI/CD

- Developed backend services using **Go and Python**, designing structured RESTful APIs for airline and booking integrations
- Improved API response efficiency by **97%** through concurrency optimization and better request orchestration
- Designed modular service components ensuring separation of concerns and maintainable backend architecture
- Integrated services with **AWS infrastructure**, managed deployments in Unix-like environments, and containerized services using Docker
- Implemented unit and integration testing pipelines to ensure production-grade reliability

</details>

<details>
<summary><b>🤖 Machine Learning Intern — Robotics and Machine Intelligence Laboratory</b> &nbsp;·&nbsp; Remote &nbsp;|&nbsp; June 2024 – July 2024</summary>

> Python · CNN · Backend ML Integration · Data Processing · Firebase · Deployment

- Built and optimized **CNN-based pipelines** in Python for medical image classification using structured preprocessing and augmentation workflows
- Developed scalable data handling modules for large genomic and imaging datasets
- Achieved **98% model accuracy** through systematic hyperparameter tuning and validation
- Integrated ML models with Firebase-backed platforms for secure storage and accessible inference workflows
- Documented system design decisions, training pipelines, and deployment steps to ensure reproducibility

</details>

---

## 📄 License

```
MIT License — Copyright (c) 2025 Ansh Sharma

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files to deal in the Software
without restriction, including the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software.
```

---

<div align="center">

**Built by [Ansh Sharma](https://www.linkedin.com/in/anshsharmacse/) · B230825MT**

[![LinkedIn](https://img.shields.io/badge/Connect_on_LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/anshsharmacse/)
[![Portfolio](https://img.shields.io/badge/View_Portfolio-39d353?style=for-the-badge&logo=linktree&logoColor=white)](https://linktr.ee/Anshsharma_21?utm_source=linktree_profile_share&ltsid=6cda2541-2501-4146-991e-cb8a5b0fecb3)
[![GitHub](https://img.shields.io/badge/Follow_on_GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/anshsharmacse/)

*PolyMind AI — Where quantum materials science meets human-centered AI*

⭐ **Star this repo** if you found it useful!

</div>
