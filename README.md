# 📅 Timeline Visualization Application

A modern, interactive timeline visualization tool built with Streamlit and Plotly. Transform your data into beautiful, interactive timeline visualizations with multiple chart types and advanced customization options.

![Timeline App](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

## ✨ Features

### 🎨 **Multiple Visualization Types**
- **📊 Timeline** - Traditional timeline with event connections
- **🕸️ Interactive Network** - Force-directed network graph
- **🔵 Scatter Plot** - Individual event points with importance mapping
- **📅 Gantt Chart** - Horizontal timeline bars for project management
- **🌊 Theme River** - Streamgraph showing subject distribution over time

### 🎛️ **Interactive Controls**
- **Detail Level Slider** - Control information density (1-10)
- **Connection Level Slider** - Adjust network complexity (1-5)
- **Zoom Controls** - Scale visualizations for detailed analysis
- **Full Screen Mode** - Immersive viewing experience

### 🎨 **Modern UI Design**
- **Dark Gradient Theme** - Professional dark mode interface
- **Glassmorphism Effects** - Modern translucent design elements
- **Responsive Layout** - Works on desktop and mobile devices
- **Custom Typography** - Clean, readable fonts throughout

### 📊 **Data Processing**
- **CSV Import** - Upload your timeline data
- **LLM Integration** - AI-powered data conversion (OpenAI/Claude)
- **Real-time Preview** - See changes instantly
- **Export Options** - Download visualizations as images

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd tarun_proj
   ```

2. **Create virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**
   ```bash
   streamlit run timeline_app.py
   ```

5. **Open your browser**
   Navigate to `http://localhost:8501`

## 📋 Data Format

### Required CSV Structure
Your timeline data should be in CSV format with the following columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `date` | Date | Event date | `2023-01-15` |
| `title` | String | Event title | `Project Launch` |
| `description` | String | Event description | `Official project launch event` |
| `importance` | Integer | Importance level (1-5) | `4` |
| `subjects` | String | Comma-separated subjects | `Technology, Business` |

### Sample Data
```csv
date,title,description,importance,subjects
2023-01-15,Project Launch,Official project launch event,4,Technology,Business
2023-02-20,First Milestone,Completed initial development phase,3,Development
2023-03-10,User Testing,Began user acceptance testing,5,Testing,Quality
```

## 🎯 Usage Guide

### 1. **Data Input**
- **Upload CSV**: Drag and drop your timeline CSV file
- **AI Conversion**: Use LLM to convert text descriptions to structured data
- **Manual Entry**: Input data directly through the form interface

### 2. **Visualization Selection**
Choose from 5 different visualization types:
- **Timeline**: Best for chronological event sequences
- **Interactive Network**: Ideal for showing relationships between events
- **Scatter Plot**: Perfect for importance vs. time analysis
- **Gantt Chart**: Great for project management and scheduling
- **Theme River**: Excellent for showing subject trends over time

### 3. **Customization**
- **Detail Level**: Adjust information density (1-10)
- **Connection Level**: Control network complexity (1-5)
- **Zoom Factor**: Scale visualizations for detailed analysis
- **Color Schemes**: Automatic color coding by importance/subject

### 4. **Interaction**
- **Hover Effects**: Rich tooltips with event details
- **Click Interactions**: Select and highlight events
- **Pan & Zoom**: Navigate through large datasets
- **Full Screen**: Immersive viewing mode

## 🛠️ Technical Architecture

### **Frontend**
- **Streamlit**: Web application framework
- **Plotly**: Interactive visualization library
- **Custom CSS**: Dark theme and glassmorphism effects

### **Backend**
- **Pandas**: Data manipulation and analysis
- **NetworkX**: Network graph creation and analysis
- **OpenAI/Anthropic**: LLM integration for data processing

### **Key Components**
```python
# Main application structure
timeline_app.py
├── TimelineVisualizer class
│   ├── create_timeline() - Main timeline creation
│   ├── _create_enhanced_interactive_network() - Network graphs
│   ├── _create_gantt_chart() - Gantt chart visualization
│   └── _create_theme_river() - Streamgraph visualization
├── Data processing functions
├── UI components and styling
└── LLM integration
```

## 🎨 Visualization Types

### 📊 **Timeline**
- **Purpose**: Traditional chronological event display
- **Features**: Event connections, importance-based sizing
- **Best For**: Historical timelines, event sequences

### 🕸️ **Interactive Network**
- **Purpose**: Show relationships between events
- **Features**: Force-directed layout, connection types
- **Best For**: Complex event relationships, causal analysis

### 🔵 **Scatter Plot**
- **Purpose**: Individual event analysis
- **Features**: Size/color by importance, hover details
- **Best For**: Event importance analysis, pattern recognition

### 📅 **Gantt Chart**
- **Purpose**: Project management timeline
- **Features**: Horizontal bars, chronological order
- **Best For**: Project planning, scheduling, resource allocation

### 🌊 **Theme River**
- **Purpose**: Subject distribution over time
- **Features**: Streamgraph visualization, subject trends
- **Best For**: Topic analysis, trend identification

## 🔧 Configuration

### Environment Variables
```bash
# OpenAI API (optional)
OPENAI_API_KEY=your_openai_api_key

# Anthropic API (optional)
ANTHROPIC_API_KEY=your_anthropic_api_key
```

### Customization Options
- **Color Schemes**: Modify `importance_colors` in the code
- **Layout Algorithms**: Change network layout algorithms
- **Styling**: Update CSS variables for theme customization

## 📦 Dependencies

### Core Dependencies
```
streamlit>=1.48.0
pandas>=2.3.1
plotly>=6.2.0
networkx>=3.2.1
matplotlib>=3.9.4
seaborn>=0.13.2
```

### Optional Dependencies
```
openai>=1.99.1      # For AI data conversion
anthropic>=0.61.0   # Alternative AI provider
```

## 🚀 Deployment

### Local Development
```bash
# Development mode with auto-reload
streamlit run timeline_app.py --server.runOnSave true
```

### Production Deployment
```bash
# Using Streamlit Cloud
# 1. Push to GitHub
# 2. Connect to Streamlit Cloud
# 3. Deploy automatically
```

### Docker Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "timeline_app.py"]
```

## 🤝 Contributing

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
3. **Commit your changes** (`git commit -m 'Add amazing feature'`)
4. **Push to the branch** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### Development Guidelines
- Follow PEP 8 style guidelines
- Add docstrings to new functions
- Include type hints for better code clarity
- Test visualizations with sample data

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Streamlit** for the amazing web app framework
- **Plotly** for interactive visualization capabilities
- **NetworkX** for network analysis tools
- **OpenAI/Anthropic** for AI integration capabilities

## 📞 Support

- **Issues**: Report bugs and feature requests on GitHub
- **Documentation**: Check the inline code comments
- **Community**: Join our discussion forum

---

**Made with ❤️ for timeline visualization enthusiasts**

*Transform your data into beautiful, interactive timelines!* 