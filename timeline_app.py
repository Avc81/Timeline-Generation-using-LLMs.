import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import re
import json
import requests
from typing import List, Dict, Any
import openai
from anthropic import Anthropic
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Configure page
st.set_page_config(
    page_title="Timeline Visualization Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

class TimelineProcessor:
    """Process and structure timeline data from LLM outputs"""
    
    def __init__(self):
        self.date_patterns = [
            r'\b(\d{4})-(\d{2})-(\d{2})\b',  # YYYY-MM-DD
            r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b',  # MM/DD/YYYY
            r'\b(\d{4})\b',  # YYYY only
            r'\b(\w+)\s+(\d{1,2}),\s+(\d{4})\b',  # Month DD, YYYY
        ]
    
    def extract_dates(self, text: str) -> List[str]:
        """Extract dates from text using regex patterns"""
        dates = []
        for pattern in self.date_patterns:
            matches = re.findall(pattern, text)
            dates.extend(matches)
        return dates
    
    def parse_timeline_text(self, text: str) -> pd.DataFrame:
        """Parse timeline text into structured DataFrame"""
        lines = text.strip().split('\n')
        events = []
        
        for line in lines:
            if '|' in line:  # Structured format
                parts = line.split('|')
                if len(parts) >= 3:
                    date_str = parts[0].strip()
                    title = parts[1].strip()
                    description = parts[2].strip() if len(parts) > 2 else ""
                    importance = parts[3].strip() if len(parts) > 3 else "3"
                    
                    events.append({
                        'date_str': date_str,
                        'title': title,
                        'description': description,
                        'importance': self._parse_importance(importance)
                    })
            else:  # Unstructured format - try to extract
                dates = self.extract_dates(line)
                if dates:
                    events.append({
                        'date_str': str(dates[0]),
                        'title': line[:100],
                        'description': line,
                        'importance': 3
                    })
        
        df = pd.DataFrame(events)
        if not df.empty:
            df['date'] = df['date_str'].apply(self._parse_date)
            df['year'] = df['date'].dt.year
            df['subjects'] = df['title'].apply(self._extract_subjects)
        
        return df
    
    def _parse_date(self, date_str: str) -> datetime:
        """Parse various date formats into datetime objects"""
        try:
            # Try different date formats
            formats = ['%Y-%m-%d', '%m/%d/%Y', '%Y', '%B %d, %Y']
            for fmt in formats:
                try:
                    return pd.to_datetime(date_str, format=fmt)
                except:
                    continue
            
            # Fallback to pandas parsing
            return pd.to_datetime(date_str, errors='coerce')
        except:
            return pd.NaT
    
    def _parse_importance(self, importance_str: str) -> int:
        """Extract importance score from string"""
        try:
            numbers = re.findall(r'\d+', importance_str)
            return int(numbers[0]) if numbers else 3
        except:
            return 3
    
    def _extract_subjects(self, title: str) -> List[str]:
        """Extract key subjects/entities from title"""
        # Simple keyword extraction - can be enhanced with NLP
        words = title.split()
        subjects = [word for word in words if word[0].isupper() and len(word) > 2]
        return subjects[:3]  # Limit to top 3 subjects

class LLMConnector:
    """Connect to various LLM APIs for timeline generation"""
    
    def __init__(self):
        self.openai_client = None
        self.anthropic_client = None
    
    def setup_openai(self, api_key: str):
        """Setup OpenAI client"""
        self.openai_client = openai.OpenAI(api_key=api_key)
    
    def setup_anthropic(self, api_key: str):
        """Setup Anthropic client"""
        self.anthropic_client = Anthropic(api_key=api_key)
    
    def generate_timeline_openai(self, prompt: str) -> str:
        """Generate timeline using OpenAI GPT"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a historian creating accurate timelines. Format your response as: Date | Event | Description | Importance (1-5)"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"
    
    def generate_timeline_anthropic(self, prompt: str) -> str:
        """Generate timeline using Anthropic Claude"""
        try:
            message = self.anthropic_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=2000,
                messages=[
                    {"role": "user", "content": f"Create a detailed timeline for: {prompt}. Format as: Date | Event | Description | Importance (1-5)"}
                ]
            )
            return message.content[0].text
        except Exception as e:
            return f"Error: {str(e)}"

class TimelineVisualizer:
    """Create various timeline visualizations"""
    
    def __init__(self):
        self.colors = px.colors.qualitative.Set3
    
    def _format_date(self, date_val):
        try:
            if pd.isna(date_val):
                return "N/A"
            return date_val.strftime('%Y-%m-%d')
        except Exception:
            return "N/A"

    def create_basic_timeline(self, df: pd.DataFrame, time_scale: str = "Linear", importance_multiplier: float = 1.0) -> go.Figure:
        """Create SCALED basic timeline visualization"""
        fig = go.Figure()
        
        # Filter out rows with invalid dates
        valid_df = df[df['date'].notna()].copy()
        
        if valid_df.empty:
            fig.add_annotation(
                text="No valid dates found in the data",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
            return fig
        
        # APPLY TIME SCALING
        if time_scale == "Logarithmic":
            min_date = valid_df['date'].min()
            days_diff = (valid_df['date'] - min_date).dt.days + 1
            x_values = np.log10(days_diff)
            x_title = "Log Scale (Days since start)"
        elif time_scale == "Decade":
            x_values = (valid_df['date'].dt.year // 10) * 10
            x_title = "Decade"
        elif time_scale == "Era-based":
            def get_era(year):
                if year < 1500: return 1
                elif year < 1800: return 2
                elif year < 1900: return 3
                elif year < 1950: return 4
                elif year < 2000: return 5
                else: return 6
            x_values = valid_df['date'].dt.year.apply(get_era)
            x_title = "Historical Era"
        else:  # Linear
            x_values = valid_df['date']
            x_title = "Date"
        
        # Create scatter plot with scaling
        for i, row in valid_df.iterrows():
            x_val = x_values[i] if time_scale != "Linear" else row['date']
            
            fig.add_trace(go.Scatter(
                x=[x_val],
                y=[row['importance']],
                mode='markers+text',
                marker=dict(
                    size=max(10, row['importance'] * 10 * importance_multiplier),
                    color=row['importance'],
                    colorscale='Viridis',
                    showscale=True if i == valid_df.index[0] else False,
                    line=dict(width=2, color='white')
                ),
                text=row['title'][:30] + "...",
                textposition="top center",
                hovertemplate=f"<b>{row['title']}</b><br>" +
                             f"Date: {self._format_date(row['date'])}<br>" +
                             f"Description: {row['description'][:100]}...<br>" +
                             f"Importance: {row['importance']}<extra></extra>",
                name=f"Event {i+1}",
                showlegend=False
            ))
        
        fig.update_layout(
            title=f"Interactive Timeline ({time_scale} Scale, {importance_multiplier}x Size)",
            xaxis_title=x_title,
            yaxis_title="Importance Level",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_theme_river(self, df: pd.DataFrame) -> go.Figure:
        """Create theme river visualization"""
        # Group by year and subjects
        yearly_data = defaultdict(lambda: defaultdict(int))
        subject_importance = defaultdict(list)
        
        for _, row in df.iterrows():
            year = row['year']
            for subject in row['subjects']:
                if subject:
                    yearly_data[year][subject] += row['importance']
                    subject_importance[subject].append(row['importance'])
        
        # Calculate average importance for each subject
        subject_avg_importance = {
            subject: np.mean(scores) 
            for subject, scores in subject_importance.items()
        }
        
        # Create river data
        years = sorted(yearly_data.keys())
        subjects = list(subject_avg_importance.keys())[:10]  # Top 10 subjects
        
        fig = go.Figure()
        
        for i, subject in enumerate(subjects):
            y_values = [yearly_data[year][subject] for year in years]
            
            fig.add_trace(go.Scatter(
                x=years,
                y=y_values,
                mode='lines',
                fill='tonexty' if i > 0 else 'tozeroy',
                line=dict(width=0),
                fillcolor=self.colors[i % len(self.colors)],
                name=subject,
                hovertemplate=f"<b>{subject}</b><br>" +
                             "Year: %{x}<br>" +
                             "Activity: %{y}<extra></extra>"
            ))
        
        fig.update_layout(
            title="Theme River Visualization",
            xaxis_title="Year",
            yaxis_title="Activity Level",
            height=600,
            hovermode='x unified'
        )
        
        return fig
    
    def create_gantt_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create Gantt-style timeline"""
        fig = go.Figure()
        
        # Sort by date
        df_sorted = df.sort_values('date')
        
        for i, row in df_sorted.iterrows():
            # Create duration (assuming 1 day for point events)
            start_date = row['date']
            end_date = start_date + timedelta(days=1) if not pd.isna(start_date) else start_date
            date_str = self._format_date(row['date'])
            fig.add_trace(go.Scatter(
                x=[start_date, end_date],
                y=[i, i],
                mode='lines+markers',
                line=dict(width=row['importance'] * 2, color=self.colors[i % len(self.colors)]),
                marker=dict(size=8),
                name=row['title'][:30],
                hovertemplate=f"<b>{row['title']}</b><br>" +
                             f"Date: {date_str}<br>" +
                             f"Importance: {row['importance']}<extra></extra>"
            ))
        
        fig.update_layout(
            title="Gantt-Style Timeline",
            xaxis_title="Date",
            yaxis_title="Events",
            height=max(400, len(df) * 30),
            showlegend=False
        )
        
        return fig
    
    def create_network_graph(self, df: pd.DataFrame) -> go.Figure:
        """Create network graph of timeline connections"""
        G = nx.Graph()
        
        # Add nodes for events
        for i, row in df.iterrows():
            G.add_node(i, 
                      title=row['title'], 
                      date=row['date'],
                      importance=row['importance'])
        
        # Add edges based on temporal proximity and subject similarity
        for i in range(len(df)):
            for j in range(i+1, len(df)):
                # Connect events close in time
                if pd.isna(df.iloc[i]['date']) or pd.isna(df.iloc[j]['date']):
                    continue
                time_diff = abs((df.iloc[i]['date'] - df.iloc[j]['date']).days)
                if time_diff < 365:  # Within 1 year
                    # Check subject overlap
                    subjects_i = set(df.iloc[i]['subjects'])
                    subjects_j = set(df.iloc[j]['subjects'])
                    overlap = len(subjects_i.intersection(subjects_j))
                    
                    if overlap > 0:
                        G.add_edge(i, j, weight=overlap)
        
        # Create layout
        pos = nx.spring_layout(G)
        
        # Extract coordinates
        x_nodes = [pos[node][0] for node in G.nodes()]
        y_nodes = [pos[node][1] for node in G.nodes()]
        
        # Create edge traces
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines'
        ))
        
        # Add nodes
        node_text = [f"{df.iloc[node]['title'][:20]}..." for node in G.nodes()]
        node_sizes = [df.iloc[node]['importance'] * 5 for node in G.nodes()]
        
        fig.add_trace(go.Scatter(
            x=x_nodes, y=y_nodes,
            mode='markers+text',
            marker=dict(size=node_sizes, color=node_sizes, colorscale='Viridis'),
            text=node_text,
            textposition="middle center",
            hoverinfo='text',
            hovertext=[f"<b>{df.iloc[node]['title']}</b><br>Date: {self._format_date(df.iloc[node]['date'])}" 
                      for node in G.nodes()]
        ))
        
        fig.update_layout(
            title="Timeline Network Graph",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[ dict(
                text="Connections show temporal and thematic relationships",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(color='gray', size=12)
            )],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig

def main():
    st.title("🕰️ Timeline Visualization Tool")
    st.markdown("Generate and visualize timelines from LLM outputs with advanced scaling and visualization options.")
    
    # Initialize components
    processor = TimelineProcessor()
    llm_connector = LLMConnector()
    visualizer = TimelineVisualizer()
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # API Configuration
    st.sidebar.subheader("LLM API Setup")
    use_llm = st.sidebar.checkbox("Use LLM Integration")
    
    if use_llm:
        llm_choice = st.sidebar.selectbox("Choose LLM", ["OpenAI GPT", "Anthropic Claude"])
        api_key = st.sidebar.text_input("API Key", type="password")
        
        if llm_choice == "OpenAI GPT" and api_key:
            llm_connector.setup_openai(api_key)
        elif llm_choice == "Anthropic Claude" and api_key:
            llm_connector.setup_anthropic(api_key)
    
    # ✨ NEW SCALING CONTROLS ✨
    st.sidebar.subheader("⚙️ Scaling Controls")
    
    time_scale = st.sidebar.selectbox(
        "Time Scale Type",
        ["Linear", "Logarithmic", "Decade", "Era-based"],
        help="Choose how time is displayed on the x-axis"
    )
    
    importance_multiplier = st.sidebar.slider(
        "Importance Size Multiplier", 
        0.5, 3.0, 1.0, 0.1,
        help="Adjust the size of timeline markers"
    )
    
    subject_filter_enabled = st.sidebar.checkbox("Enable Subject Filtering")
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Timeline Input")
        
        input_method = st.radio("Input Method", ["Manual Text", "LLM Generation", "File Upload"])
        
        timeline_data = None
        
        if input_method == "Manual Text":
            timeline_text = st.text_area(
                "Enter timeline data",
                height=300,
                placeholder="Format: Date | Event | Description | Importance\n1969-07-20 | Moon Landing | Apollo 11 lands on moon | 5"
            )
            if timeline_text:
                timeline_data = processor.parse_timeline_text(timeline_text)
        
        elif input_method == "LLM Generation" and use_llm and api_key:
            prompt = st.text_area(
                "Enter timeline prompt",
                height=100,
                placeholder="e.g., Create a timeline of the Space Race from 1957 to 1975"
            )
            
            if st.button("Generate Timeline"):
                with st.spinner("Generating timeline..."):
                    if llm_choice == "OpenAI GPT":
                        response = llm_connector.generate_timeline_openai(prompt)
                    else:
                        response = llm_connector.generate_timeline_anthropic(prompt)
                    
                    st.text_area("LLM Response", response, height=200)
                    timeline_data = processor.parse_timeline_text(response)
        
        elif input_method == "File Upload":
            uploaded_file = st.file_uploader("Upload timeline file", type=['txt', 'csv'])
            if uploaded_file:
                if uploaded_file.name.endswith('.csv'):
                    timeline_data = pd.read_csv(uploaded_file)
                    # Parse 'date' column if it exists
                    if 'date' in timeline_data.columns:
                        timeline_data['date'] = pd.to_datetime(timeline_data['date'], errors='coerce')
                    else:
                        timeline_data['date'] = pd.NaT
                    # Fill missing columns for compatibility
                    if 'title' not in timeline_data.columns:
                        timeline_data['title'] = ''
                    if 'description' not in timeline_data.columns:
                        timeline_data['description'] = ''
                    if 'importance' not in timeline_data.columns:
                        timeline_data['importance'] = 3
                    if 'subjects' not in timeline_data.columns:
                        timeline_data['subjects'] = [[] for _ in range(len(timeline_data))]
                    # Extract year for theme river
                    timeline_data['year'] = timeline_data['date'].dt.year
                else:
                    timeline_text = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
                    timeline_data = processor.parse_timeline_text(timeline_text)
    
    with col2:
        if timeline_data is not None and not timeline_data.empty:
            # ✨ NEW SUBJECT FILTERING ✨
            if subject_filter_enabled:
                st.subheader("🎯 Filter by Subject")
                
                # Extract all subjects from titles
                all_subjects = []
                for _, row in timeline_data.iterrows():
                    if 'title' in row:
                        words = str(row['title']).split()
                        subjects = [word for word in words if len(word) > 2 and word[0].isupper()]
                        all_subjects.extend(subjects)
                
                unique_subjects = list(set(all_subjects))
                
                if unique_subjects:
                    selected_subjects = st.multiselect(
                        "Select subjects to show",
                        unique_subjects,
                        default=[]
                    )
                    
                    # Filter timeline data
                    if selected_subjects:
                        def has_subject(row):
                            title_words = str(row.get('title', '')).split()
                            title_subjects = [word for word in title_words if len(word) > 2 and word[0].isupper()]
                            return any(subj in title_subjects for subj in selected_subjects)
                        
                        mask = timeline_data.apply(has_subject, axis=1)
                        timeline_data = timeline_data[mask]
                        
                        if timeline_data.empty:
                            st.warning("No events match selected subjects")
            
            st.subheader("Timeline Visualizations")
            
            # Display data summary
            st.metric("Total Events", len(timeline_data))
            
            # Check for valid dates before displaying the range
            if timeline_data['date'].notna().any():
                date_range = f"{timeline_data['date'].min().strftime('%Y-%m-%d')} to {timeline_data['date'].max().strftime('%Y-%m-%d')}"
                st.metric("Date Range", date_range)
            else:
                st.metric("Date Range", "N/A - No valid dates found")
            
            # Visualization options
            viz_type = st.selectbox(
                "Visualization Type",
                ["Basic Timeline", "Theme River", "Gantt Chart", "Network Graph"]
            )
            
            # Create visualization based on selection
            if viz_type == "Basic Timeline":
                fig = visualizer.create_basic_timeline(timeline_data, time_scale, importance_multiplier)
            elif viz_type == "Theme River":
                fig = visualizer.create_theme_river(timeline_data)
            elif viz_type == "Gantt Chart":
                fig = visualizer.create_gantt_chart(timeline_data)
            elif viz_type == "Network Graph":
                fig = visualizer.create_network_graph(timeline_data)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Additional analysis
            st.subheader("Timeline Analysis")
            
            col2a, col2b = st.columns(2)
            
            with col2a:
                # Importance distribution
                fig_importance = px.histogram(
                    timeline_data, 
                    x='importance', 
                    title="Event Importance Distribution",
                    nbins=5
                )
                st.plotly_chart(fig_importance, use_container_width=True)
            
            with col2b:
                # Events over time
                timeline_data['year'] = timeline_data['date'].dt.year
                yearly_counts = timeline_data.groupby('year').size().reset_index(name='count')
                fig_yearly = px.line(
                    yearly_counts, 
                    x='year', 
                    y='count', 
                    title="Events Over Time"
                )
                st.plotly_chart(fig_yearly, use_container_width=True)
            
            # Data table
            st.subheader("Timeline Data")
            st.dataframe(timeline_data[['date', 'title', 'description', 'importance']], use_container_width=True)
            
            # Export options
            st.subheader("Export Options")
            col3a, col3b, col3c = st.columns(3)
            
            with col3a:
                csv = timeline_data.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="timeline_data.csv",
                    mime="text/csv"
                )
            
            with col3b:
                json_data = timeline_data.to_json(orient='records', date_format='iso')
                st.download_button(
                    label="Download JSON",
                    data=json_data,
                    file_name="timeline_data.json",
                    mime="application/json"
                )
            
            with col3c:
                # Export visualization as HTML
                fig_html = fig.to_html()
                st.download_button(
                    label="Download Visualization",
                    data=fig_html,
                    file_name="timeline_visualization.html",
                    mime="text/html"
                )
        
        else:
            st.info("Please provide timeline data using one of the input methods on the left.")
            
            # Show scaling features info
            st.subheader("🚀 New Scaling Features")
            st.markdown("""
            **Time Scaling Options:**
            - **Linear**: Equal spacing between dates
            - **Logarithmic**: Compressed for long historical periods
            - **Decade**: Group events by 10-year periods
            - **Era-based**: Historical era grouping (Ancient, Medieval, Modern, etc.)
            
            **Importance Scaling:**
            - Adjust marker sizes with the multiplier slider
            - Larger values make important events more prominent
            
            **Subject Filtering:**
            - Enable to filter events by specific topics
            - Select multiple subjects to focus analysis
            
            **Try with sample data:**
            ```
            1947-12-23 | Transistor Invented | Bell Labs invents transistor | 5
            1969-07-20 | Moon Landing | Apollo 11 lands on moon | 5
            2022-11-30 | ChatGPT Launch | AI chatbot goes mainstream | 5
            ```
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("**Timeline Visualization Tool** - Built with Streamlit, Plotly, and LLM APIs")
    st.markdown("**✨ Now with Advanced Scaling Features**: Time scaling, importance multipliers, and subject filtering!")

if __name__ == "__main__":
    main()