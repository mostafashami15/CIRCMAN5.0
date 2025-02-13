Here's the complete content for `docs/diagrams/architecture.md`:

```markdown
# CIRCMAN5.0 System Architecture Diagrams

## System Overview
```plaintext
CIRCMAN5.0 System Architecture
┌─────────────────────────────┐
│      Human Interface        │
├─────────────────────────────┤
│    ┌───────────────┐        │
│    │   Dashboard   │        │
│    └───────────────┘        │
└─────────────┬───────────────┘
              │
┌─────────────┼───────────────┐
│   Digital Twin System       │
├─────────────────────────────┤
│    ┌───────────────┐        │
│    │  Simulation   │        │
│    └───────────────┘        │
└─────────────┬───────────────┘
              │
┌─────────────┼───────────────┐
│   Manufacturing System      │
└─────────────────────────────┘
```

## Component Architecture

### Manufacturing System
```plaintext
Manufacturing System
├── Process Control
│   ├── Batch Tracking
│   ├── Quality Control
│   └── Resource Management
├── Monitoring
│   ├── Real-time Data
│   ├── Performance Metrics
│   └── Alert System
└── Analysis
    ├── Efficiency
    ├── Quality
    └── Sustainability
```

### Digital Twin
```plaintext
Digital Twin System
├── State Management
│   ├── Synchronization
│   ├── Validation
│   └── History
├── Simulation
│   ├── Process Models
│   ├── Optimization
│   └── What-if Analysis
└── Visualization
    ├── 3D View
    ├── Real-time Updates
    └── Interactive Controls
```

### Human Interface
```plaintext
Human Interface System
├── Operator Dashboard
│   ├── Process View
│   ├── Controls
│   └── Alerts
├── Training System
│   ├── Tutorials
│   ├── Simulations
│   └── Assessment
└── Decision Support
    ├── Recommendations
    ├── Analytics
    └── Reports
```

### AI/ML Pipeline
```plaintext
AI System
├── Data Pipeline
│   ├── Collection
│   ├── Preprocessing
│   └── Validation
├── Model Training
│   ├── Feature Engineering
│   ├── Training
│   └── Validation
└── Deployment
    ├── Prediction
    ├── Optimization
    └── Monitoring
```

### LCA System
```plaintext
LCA System
├── Impact Analysis
│   ├── Manufacturing
│   ├── Use Phase
│   └── End-of-Life
├── Resource Tracking
│   ├── Materials
│   ├── Energy
│   └── Water
└── Circularity
    ├── Metrics
    ├── Optimization
    └── Reporting
```

## Data Flow
```plaintext
Data Flow Architecture
┌──────────┐    ┌──────────┐    ┌──────────┐
│ Sensors  │ -> │ Digital  │ -> │ Analysis │
└──────────┘    │  Twin    │    └──────────┘
                └──────────┘
                     │
              ┌──────┴───────┐
              │ Human        │
              │ Interface    │
              └──────────────┘
```

## Integration Points
```plaintext
System Integration
├── External Systems
│   ├── ERP
│   ├── MES
│   └── SCADA
├── Data Exchange
│   ├── APIs
│   ├── Protocols
│   └── Formats
└── Security
    ├── Authentication
    ├── Authorization
    └── Encryption
```

## Deployment Architecture
```plaintext
Deployment Structure
├── Production
│   ├── Main Server
│   ├── Database
│   └── Processing Units
├── Development
│   ├── Test Environment
│   ├── CI/CD Pipeline
│   └── Version Control
└── Monitoring
    ├── Performance
    ├── Security
    └── Logging
```
