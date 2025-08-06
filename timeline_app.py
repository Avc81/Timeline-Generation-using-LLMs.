import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import re
import networkx as nx
from typing import List, Dict, Tuple
import os
try:
    import openai
except ImportError:
    openai = None
try:
    import anthropic
except ImportError:
    anthropic = None

# Configure page
st.set_page_config(
    page_title="Timeline Visualizer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'timeline_data' not in st.session_state:
    st.session_state.timeline_data = None
if 'manual_events' not in st.session_state:
    st.session_state.manual_events = []

class SimpleTimelineConverter:
    """Simple, reliable timeline converter using template-guided approach"""
    
    def __init__(self):
        self.importance_colors = {
            1: '#94a3b8', 2: '#fbbf24', 3: '#60a5fa', 4: '#34d399', 5: '#f87171'
        }
    
    def get_llm_template(self, topic: str = "", time_period: str = "") -> str:
        """Generate LLM template for perfect output"""
        return f"""Create a timeline for {topic if topic else '[YOUR TOPIC]'} using this EXACT format:

YYYY-MM-DD | Event Title | Brief Description | Importance(1-5) | Subjects

RULES:
- One event per line
- Use YYYY-MM-DD format (e.g., 1969-07-20)
- Keep titles under 50 characters
- Keep descriptions under 150 characters
- Rate importance: 1=minor, 5=major historical significance
- NO emojis, bullets, or extra formatting
- List one or more subjects (comma-separated) for each event (required for Theme River)

Time period: {time_period if time_period else '[START YEAR] to [END YEAR]'}
Number of events: 10-15

EXAMPLE FORMAT:
1969-07-20 | Moon Landing | Apollo 11 successfully lands on lunar surface | 5 | Space Race, World War II
1961-04-12 | First Human in Space | Yuri Gagarin completes orbit of Earth | 5 | Space Exploration

START YOUR RESPONSE HERE:"""
    
    def parse_template_response(self, text: str) -> List[Dict]:
        """Parse LLM response that follows the template"""
        events = []
        lines = text.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#') or '|' not in line:
                continue
            
            parts = [part.strip() for part in line.split('|')]
            if len(parts) >= 4:
                try:
                    date_str = parts[0]
                    title = parts[1][:50]
                    description = parts[2][:150]
                    importance = int(parts[3])
                    subjects = parts[4] if len(parts) > 4 else ""
                    
                    if re.match(r'\d{4}-\d{2}-\d{2}', date_str):
                        parsed_date = pd.to_datetime(date_str)
                    else:
                        year_match = re.search(r'\b(\d{4})\b', date_str)
                        if year_match:
                            parsed_date = pd.to_datetime(f"{year_match.group(1)}-06-01")
                        else:
                            continue
                    
                    events.append({
                        'date': parsed_date,
                        'title': title,
                        'description': description,
                        'importance': max(1, min(5, importance)),
                        'subjects': subjects
                    })
                except:
                    continue
            
        return events
    
    def smart_parse_llm_output(self, text: str) -> list:
        events = []
        lines = text.strip().split('\n')
        date_pattern = r'(\d{4}-\d{2}-\d{2})'
        
        for line in lines:
            if not line.strip() or line.strip().startswith('#'):
                continue
            
            date_match = re.search(date_pattern, line)
            if not date_match:
                continue
            
            date_str = date_match.group(1)
            rest = line.replace(date_str, '', 1).strip(' |:-\t')
            parts = re.split(r'\s*\|\s*|\s*-\s*|\s*:\s*|\t+', rest)
            
            title = parts[0] if len(parts) > 0 else ''
            description = parts[1] if len(parts) > 1 else ''
            importance = 3
            
            for p in parts:
                if p.strip().isdigit() and 1 <= int(p.strip()) <= 5:
                    importance = int(p.strip())
                    break
            
            subjects = parts[-1] if len(parts) > 2 else ''
            
            try:
                parsed_date = pd.to_datetime(date_str)
                events.append({
                    'date': parsed_date,
                    'title': title.strip()[:50],
                    'description': description.strip()[:150],
                    'importance': importance,
                    'subjects': subjects.strip()
                })
            except Exception:
                continue
        
        return events
    
    def create_dataframe(self, events: List[Dict]) -> pd.DataFrame:
        """Create DataFrame from events"""
        if not events:
            return pd.DataFrame()
        
        df = pd.DataFrame(events)
        df = df.sort_values('date', na_position='last').reset_index(drop=True)
        df['year'] = df['date'].dt.year
        return df
    
    def to_standard_format(self, df: pd.DataFrame) -> str:
        """Convert DataFrame to standard format string"""
        if df.empty:
            return "# No events found"
        
        lines = [
            "# Standard Timeline Format",
            "# Format: YYYY-MM-DD | Event Title | Description | Importance (1-5)",
            ""
        ]
        
        for _, row in df.iterrows():
            date_str = row['date'].strftime('%Y-%m-%d')
            line = f"{date_str} | {row['title']} | {row['description']} | {row['importance']}"
            lines.append(line)
        
        return "\n".join(lines)

class TimelineVisualizer:
    """Enhanced timeline visualizer with interactive network graph"""
    
    def __init__(self):
        self.importance_colors = {
            1: '#94a3b8', 2: '#fbbf24', 3: '#60a5fa', 4: '#34d399', 5: '#f87171'
        }
    
    def create_timeline(self, df: pd.DataFrame, chart_type: str = "Timeline", size_multiplier: float = 1.0, group_by: str = None, detail_level: int = 5) -> go.Figure:
        """Create timeline visualization with zoom and detail support"""
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data to display", x=0.5, y=0.5, xref="paper", yref="paper", font=dict(size=16))
            return fig
        
        # Main function logic
        if chart_type == "Timeline":
            return self._create_enhanced_timeline(df, size_multiplier, detail_level)
        elif chart_type == "Interactive Network":
            return self._create_enhanced_interactive_network(df)
        elif chart_type == "Theme River":
            return self._create_theme_river(df, group_by, size_multiplier, detail_level)
        elif chart_type == "Heatmap":
            return self._create_event_heatmap(df, detail_level)
        else:
            return self._create_enhanced_timeline(df, size_multiplier, detail_level)

    def _create_enhanced_timeline(self, df: pd.DataFrame, size_multiplier: float, detail_level: int = 5) -> go.Figure:
        """Enhanced timeline with smart positioning"""
        fig = go.Figure()
        
        n_events = len(df)
        y_positions = []
        
        for i in range(n_events):
            if n_events <= 8:
                height_levels = [1, 3, 2, 4, 1.5, 3.5, 2.5, 4.5]
                y_pos = height_levels[i % len(height_levels)]
            else:
                base_height = 2.5
                wave_offset = np.sin(i * 0.8) * 1.5
                y_pos = base_height + wave_offset
            
            y_positions.append(y_pos)
        
        subject_colors = None
        if 'subjects' in df.columns:
            all_subjects = set()
            for s in df['subjects'].fillna(""):
                all_subjects.update([x.strip() for x in str(s).split(',') if x.strip()])
            palette = px.colors.qualitative.Plotly + px.colors.qualitative.Pastel
            subject_colors = {s: palette[i % len(palette)] for i, s in enumerate(sorted(all_subjects))}
        
        for i, (_, row) in enumerate(df.iterrows()):
            color = self.importance_colors.get(row['importance'], '#60a5fa')
            
            if subject_colors and 'subjects' in row and row['subjects']:
                first_subject = str(row['subjects']).split(',')[0].strip()
                if first_subject in subject_colors:
                    color = subject_colors[first_subject]
            
            size = max(15, row['importance'] * 10 * size_multiplier)
            
            if detail_level >= 8:
                hover_text = f"<b>{row['title']}</b><br>"
                hover_text += f"Date: {row['date'].strftime('%B %d, %Y')}<br>"
                hover_text += f"Description: {row['description']}<br>"
                hover_text += f"Importance: {row['importance']}/5"
                if subject_colors and 'subjects' in row and row['subjects']:
                    hover_text += f"<br>Subjects: {row['subjects']}"
            elif detail_level >= 5:
                hover_text = f"<b>{row['title']}</b><br>"
                hover_text += f"Date: {row['date'].strftime('%Y-%m-%d')}<br>"
                hover_text += f"Importance: {row['importance']}/5"
            else:
                hover_text = f"<b>{row['title']}</b><br>"
                hover_text += f"Date: {row['date'].strftime('%Y')}"
            
            fig.add_trace(go.Scatter(
                x=[row['date']],
                y=[y_positions[i]],
                mode='markers+text',
                marker=dict(size=size, color=color, line=dict(width=2, color='white')),
                text=row['title'][:40] if detail_level >= 3 else "",
                textfont=dict(size=12, color='#ffffff'),
                textposition="top center" if i % 2 == 0 else "bottom center",
                hovertemplate=hover_text + "<extra></extra>",
                showlegend=False,
                name=row['title']
            ))
        
            if i > 0:
                fig.add_trace(go.Scatter(
                    x=[df.iloc[i-1]['date'], row['date']],
                    y=[y_positions[i-1], y_positions[i]],
                    mode='lines',
                    line=dict(width=1.5, color='rgba(148, 163, 184, 0.4)', dash='dot'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Add legend for importance levels
        legend_traces = []
        for importance in sorted(df['importance'].unique()):
            color = self.importance_colors.get(importance, '#60a5fa')
            legend_traces.append(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=color),
                name=f'Importance {importance}/5',
                showlegend=True
            ))
        
        # Add subject legend if available
        if subject_colors:
            for subject, color in subject_colors.items():
                legend_traces.append(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(size=10, color=color),
                    name=f'Subject: {subject}',
                    showlegend=True
                ))
        
        # Add connection line legend
        legend_traces.append(go.Scatter(
            x=[None], y=[None],
            mode='lines',
            line=dict(width=1.5, color='rgba(148, 163, 184, 0.4)', dash='dot'),
            name='Event Connections',
            showlegend=True
        ))
        
        # Add all legend traces to the figure
        for trace in legend_traces:
            fig.add_trace(trace)
        
        fig.update_layout(
            title=dict(text="Timeline Visualization", font=dict(size=20, color='#ffffff')),
            xaxis_title="Date",
            yaxis_title="Events",
            height=500,
            hovermode='closest',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', size=12),
            legend=dict(
                bgcolor='rgba(0,0,0,0.7)',
                bordercolor='rgba(255,255,255,0.3)',
                borderwidth=1,
                font=dict(color='#ffffff', size=10)
            )
        )
        
        return fig
    
    def _create_enhanced_interactive_network(self, df: pd.DataFrame, detail_level: int = 5, connection_level: int = 3) -> go.Figure:
        """Interactive network graph"""
        if len(df) < 2:
            fig = go.Figure()
            fig.add_annotation(text="Need at least 2 events for network view", 
                             x=0.5, y=0.5, xref="paper", yref="paper")
            return fig
        
        G = nx.Graph()
        
        for i, row in df.iterrows():
            G.add_node(i, 
                      title=row['title'],
                      date=row['date'],
                      importance=row['importance'],
                      description=row['description'],
                      subjects=row.get('subjects', ''),
                      year=row['year'])
        
        # Add temporal connections - connect all sequential events
        for i in range(len(df) - 1):
            time_diff = abs((df.iloc[i+1]['date'] - df.iloc[i]['date']).days)
            weight = max(0.1, 1.0 / (1 + time_diff / 365))
            G.add_edge(i, i+1, weight=weight, type='temporal')
        
        # Add subject connections based on connection level
        if 'subjects' in df.columns:
            for i in range(len(df)):
                if df.iloc[i]['subjects']:
                    subjects_i = set([s.strip().lower() for s in str(df.iloc[i]['subjects']).split(',') if s.strip()])
                    for j in range(i+1, len(df)):
                        if df.iloc[j]['subjects']:
                            subjects_j = set([s.strip().lower() for s in str(df.iloc[j]['subjects']).split(',') if s.strip()])
                            overlap = len(subjects_i & subjects_j)
                            if overlap > 0:
                                weight = overlap / max(len(subjects_i), len(subjects_j))
                                # More lenient threshold based on connection level
                                threshold = max(0.1, 0.6 - (connection_level * 0.1))
                                if weight >= threshold:
                                    G.add_edge(i, j, weight=weight, type='subject')
        
        # Add importance-based connections for higher connection levels
        if connection_level >= 3:
            for i in range(len(df)):
                for j in range(i+1, len(df)):
                    importance_diff = abs(df.iloc[i]['importance'] - df.iloc[j]['importance'])
                    if importance_diff <= 1:  # Connect events with similar importance
                        weight = 0.3 + (5 - importance_diff) * 0.1
                        G.add_edge(i, j, weight=weight, type='importance')
        
        # Ensure minimum connectivity - connect each node to at least one other
        for node in G.nodes():
            if len(list(G.neighbors(node))) == 0:
                # Find closest node by date
                closest_node = None
                min_time_diff = float('inf')
                for other_node in G.nodes():
                    if other_node != node:
                        time_diff = abs((df.iloc[node]['date'] - df.iloc[other_node]['date']).days)
                        if time_diff < min_time_diff:
                            min_time_diff = time_diff
                            closest_node = other_node
                if closest_node is not None:
                    G.add_edge(node, closest_node, weight=0.2, type='temporal')
        
        pos = nx.spring_layout(G, k=5, iterations=150, seed=42)
        
        edge_traces = []
        edge_colors = {
            'temporal': '#f87171',
            'subject': '#34d399',
            'importance': '#60a5fa'
        }
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = edge[2].get('weight', 0.5)
            edge_type = edge[2].get('type', 'temporal')
            
            color = edge_colors.get(edge_type, '#94a3b8')
            # Make edges more visible
            width = max(2, weight * 12)
            
            edge_traces.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=width, color=color),
                hoverinfo='none',
                showlegend=False,
                opacity=0.8
            ))
        
        node_x, node_y, node_text, node_hover, node_sizes, node_colors = [], [], [], [], [], []
        
        all_subjects = set()
        for _, row in df.iterrows():
            if row.get('subjects'):
                all_subjects.update([s.strip() for s in str(row['subjects']).split(',') if s.strip()])
        
        subject_colors = {}
        colors = ['#f87171', '#34d399', '#60a5fa', '#fbbf24', '#a78bfa', '#fb7185', '#34d399', '#fbbf24']
        for i, subject in enumerate(sorted(all_subjects)):
            subject_colors[subject] = colors[i % len(colors)]
        
        for node in G.nodes():
            x, y = pos[node]
            row = df.iloc[node]
            
            node_x.append(x)
            node_y.append(y)
            node_text.append(row['title'][:15])
            
            hover_text = f"<b>{row['title']}</b><br>"
            hover_text += f"Date: {row['date'].strftime('%Y-%m-%d')}<br>"
            hover_text += f"Importance: {row['importance']}/5<br>"
            hover_text += f"Description: {row['description'][:80]}"
            if row.get('subjects'):
                hover_text += f"<br>Subjects: {row['subjects']}"
            node_hover.append(hover_text)
            
            connections = len(list(G.neighbors(node)))
            base_size = 25 + (row['importance'] - 1) * 10
            # Adjust size based on detail level and ensure minimum visibility
            detail_multiplier = detail_level / 5.0
            size = max(20, (base_size + connections * 3) * detail_multiplier)
            node_sizes.append(size)
            
            if row.get('subjects'):
                first_subject = str(row['subjects']).split(',')[0].strip()
                if first_subject in subject_colors:
                    node_colors.append(subject_colors[first_subject])
                else:
                    node_colors.append(self.importance_colors.get(row['importance'], '#60a5fa'))
            else:
                node_colors.append(self.importance_colors.get(row['importance'], '#60a5fa'))
        
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers+text',
            text=node_text,
                textposition="top center",
            textfont=dict(size=10, color='white'),
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='white'),
                opacity=0.9,
                symbol='circle'
            ),
            hovertemplate="%{hovertext}<extra></extra>",
            hovertext=node_hover,
                showlegend=False
        )
        
        # Add legend traces for network elements
        legend_traces = []
        
        # Edge type legend
        edge_colors = {
            'temporal': '#f87171',
            'subject': '#34d399',
            'importance': '#60a5fa'
        }
        
        for edge_type, color in edge_colors.items():
            legend_traces.append(go.Scatter(
                x=[None], y=[None],
                mode='lines',
                line=dict(width=3, color=color),
                name=f'{edge_type.title()} Connections',
                showlegend=True
            ))
        
        # Node size legend
        legend_traces.append(go.Scatter(
            x=[None], y=[None],
            mode='markers',
            marker=dict(size=15, color='#60a5fa', line=dict(width=2, color='white')),
            name='Events (size = importance + connections)',
            showlegend=True
        ))
        
        # Subject legend if available
        if subject_colors:
            for subject, color in subject_colors.items():
                legend_traces.append(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(size=10, color=color),
                    name=f'Subject: {subject}',
                    showlegend=True
                ))
        
        fig = go.Figure(data=edge_traces + [node_trace] + legend_traces)
        
        fig.update_layout(
            title=dict(text="Interactive Network Graph", font=dict(size=20, color='#ffffff')),
            height=500,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.3, 1.3]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1.3, 1.3]),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', size=12),
            margin=dict(l=0, r=0, t=50, b=0),
            hovermode='closest',
            legend=dict(
                bgcolor='rgba(0,0,0,0.7)',
                bordercolor='rgba(255,255,255,0.3)',
                borderwidth=1,
                font=dict(color='#ffffff', size=10),
                x=0.02, y=0.98,
                xanchor='left', yanchor='top'
            )
        )
        
        return fig
    
    def _create_event_scatter_plot(self, df: pd.DataFrame, detail_level: int = 5) -> go.Figure:
        """Scatter plot showing individual events with size and color based on importance"""
        if df.empty:
            return go.Figure()
        
        df['year'] = df['date'].dt.year
        
        # Create color mapping for subjects if available
        subject_colors = None
        if 'subjects' in df.columns:
            all_subjects = set()
            for s in df['subjects'].fillna(""):
                all_subjects.update([x.strip() for x in str(s).split(',') if x.strip()])
            palette = px.colors.qualitative.Plotly + px.colors.qualitative.Pastel
            subject_colors = {s: palette[i % len(palette)] for i, s in enumerate(sorted(all_subjects))}
        
        fig = go.Figure()
        
        for i, (_, row) in enumerate(df.iterrows()):
            color = self.importance_colors.get(row['importance'], '#60a5fa')
            
            if subject_colors and 'subjects' in row and row['subjects']:
                first_subject = str(row['subjects']).split(',')[0].strip()
                if first_subject in subject_colors:
                    color = subject_colors[first_subject]
            
            size = max(10, row['importance'] * 8)
            
            if detail_level >= 8:
                hover_text = f"<b>{row['title']}</b><br>Date: {row['date'].strftime('%B %d, %Y')}<br>Importance: {row['importance']}/5<br>Description: {row['description'][:100]}"
            else:
                hover_text = f"<b>{row['title']}</b><br>Date: {row['date'].strftime('%Y-%m-%d')}<br>Importance: {row['importance']}/5"
            
            fig.add_trace(go.Scatter(
                x=[row['year']],
                y=[row['importance']],
                mode='markers',
                marker=dict(
                    size=size,
                    color=color,
                    opacity=0.8,
                    line=dict(width=2, color='white')
                ),
                text=row['title'][:30] if detail_level >= 5 else "",
                textposition="top center",
                textfont=dict(size=10, color='white'),
                hovertemplate=hover_text + "<extra></extra>",
                showlegend=False,
                name=row['title']
            ))
        
        fig.update_layout(
            title=dict(text="Event Scatter Plot", font=dict(size=20, color='#ffffff')),
            xaxis_title="Year",
            yaxis_title="Importance Level",
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', size=12)
        )
        
        return fig

    def _create_gantt_chart(self, df: pd.DataFrame, detail_level: int = 5) -> go.Figure:
        """Gantt chart showing events as horizontal bars over time"""
        if df.empty:
            return go.Figure()
        
        # Sort events by date
        df_sorted = df.sort_values('date').reset_index(drop=True)
        
        # Create color mapping for subjects if available
        subject_colors = None
        if 'subjects' in df.columns:
            all_subjects = set()
            for s in df['subjects'].fillna(""):
                all_subjects.update([x.strip() for x in str(s).split(',') if x.strip()])
            palette = px.colors.qualitative.Plotly + px.colors.qualitative.Pastel
            subject_colors = {s: palette[i % len(palette)] for i, s in enumerate(sorted(all_subjects))}
        
        fig = go.Figure()
        
        for i, (_, row) in enumerate(df_sorted.iterrows()):
            # Determine color based on importance or subject
            color = self.importance_colors.get(row['importance'], '#60a5fa')
            
            if subject_colors and 'subjects' in row and row['subjects']:
                first_subject = str(row['subjects']).split(',')[0].strip()
                if first_subject in subject_colors:
                    color = subject_colors[first_subject]
            
            # Create hover text
            if detail_level >= 8:
                hover_text = f"<b>{row['title']}</b><br>Date: {row['date'].strftime('%B %d, %Y')}<br>Importance: {row['importance']}/5<br>Description: {row['description'][:100]}"
            else:
                hover_text = f"<b>{row['title']}</b><br>Date: {row['date'].strftime('%Y-%m-%d')}<br>Importance: {row['importance']}/5"
            
            # Add event as a horizontal bar
            fig.add_trace(go.Bar(
                y=[f"{row['title'][:40]}{'...' if len(row['title']) > 40 else ''}"],
                x=[1],  # Width of 1 day
                base=row['date'],
                orientation='h',
                marker=dict(
                    color=color,
                    opacity=0.8,
                    line=dict(width=1, color='white')
                ),
                hovertemplate=hover_text + "<extra></extra>",
                showlegend=False,
                name=row['title']
            ))
        
        fig.update_layout(
            title=dict(text="Timeline Gantt Chart", font=dict(size=20, color='#ffffff')),
            xaxis_title="Date",
            yaxis_title="Events",
            height=max(400, len(df) * 30),  # Adjust height based on number of events
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', size=12),
            barmode='overlay',
            xaxis=dict(
                type='date',
                tickformat='%Y-%m-%d',
                tickangle=45
            ),
            yaxis=dict(
                autorange='reversed'  # Show events in chronological order
            )
        )
        
        return fig

    def _create_theme_river(self, df: pd.DataFrame, group_by: str = None, size_multiplier: float = 1.0, detail_level: int = 5) -> go.Figure:
        """Theme River visualization"""
        if 'subjects' not in df.columns:
            fig = go.Figure()
            fig.add_annotation(text="No subject data for Theme River", x=0.5, y=0.5, xref="paper", yref="paper", font=dict(size=16))
            return fig
        
        df = df.copy()
        df['subjects_list'] = df['subjects'].fillna("").apply(lambda x: [s.strip() for s in str(x).split(',') if s.strip()])
        df_exploded = df.explode('subjects_list')
        
        if df_exploded['subjects_list'].isna().all() or df_exploded['subjects_list'].eq('').all():
            fig = go.Figure()
            fig.add_annotation(text="No subject data for Theme River", x=0.5, y=0.5, xref="paper", yref="paper", font=dict(size=16))
            return fig
        
        if not group_by:
            group_by = "Year"
        
        if group_by == "Year":
            df_exploded['group'] = df_exploded['date'].dt.year
            x_title = "Year"
        elif group_by == "Month":
            df_exploded['group'] = df_exploded['date'].dt.to_period('M').astype(str)
            x_title = "Month"
        elif group_by == "Day":
            df_exploded['group'] = df_exploded['date'].dt.strftime('%Y-%m-%d')
            x_title = "Day"
        else:
            df_exploded['group'] = df_exploded['date'].dt.year
            x_title = "Year"
        
        grouped = df_exploded.groupby(['group', 'subjects_list'])['importance'].sum().reset_index()
        pivot = grouped.pivot(index='group', columns='subjects_list', values='importance').fillna(0)
        pivot = pivot.sort_index(axis=1)
        
        fig = go.Figure()
        colors = px.colors.qualitative.Plotly + px.colors.qualitative.Pastel
        
        for i, subject in enumerate(pivot.columns):
            fig.add_trace(go.Scatter(
                x=pivot.index,
                y=pivot[subject],
                mode='lines',
                stackgroup='one',
                name=subject,
                line=dict(width=0.5),
                fillcolor=colors[i % len(colors)],
                hovertemplate=f"<b>{subject}</b><br>{x_title}: %{{x}}<br>Importance: %{{y}}<extra></extra>"
            ))
        
        fig.update_layout(
            title=dict(text="Theme River (Subject Streamgraph)", font=dict(size=20, color='#ffffff')),
            xaxis_title=x_title,
            yaxis_title="Total Importance",
            height=500,
            hovermode='x unified',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#ffffff', size=12),
            legend=dict(
                bgcolor='rgba(0,0,0,0.7)',
                bordercolor='rgba(255,255,255,0.3)',
                borderwidth=1,
                font=dict(color='#ffffff', size=10),
                title=dict(text="Subjects", font=dict(color='#ffffff', size=12))
            )
        )
        return fig

def get_llm_timeline(prompt, provider, api_key):
    if provider == "ChatGPT" and openai:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1200,
            temperature=0.7,
        )
        return response.choices[0].message.content
    elif provider == "Claude" and anthropic:
        try:
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-3-opus-20240229",
                max_tokens=1200,
                temperature=0.7,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except anthropic.BadRequestError as e:
            if 'credit balance is too low' in str(e).lower():
                return "[Anthropic API: Credit balance too low. Please add credits to your account.]"
            else:
                return f"[Anthropic API error: {e}]"
    else:
        return "[LLM provider not available or not installed]"

def main():
    """Main application with modern dashboard UI"""
    
    # Modern dashboard CSS
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Product+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Product Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 25%, #0f3460 50%, #533483 75%, #7209b7 100%);
        min-height: 100vh;
        font-family: 'Product Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 25%, #0f3460 50%, #533483 75%, #7209b7 100%);
        font-family: 'Product Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* Fix text visibility */
    .stMarkdown, .stMarkdown p, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: #ffffff !important;
        font-family: 'Product Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    .stText {
        color: #ffffff !important;
        font-family: 'Product Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* Header */
    .dashboard-header {
        background: rgba(255, 255, 255, 0.1);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 2rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .dashboard-title {
        font-size: 2.5rem;
        font-weight: 600;
        color: #ffffff !important;
        margin: 0;
        line-height: 1.2;
        font-family: 'Product Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    .dashboard-subtitle {
        color: #e2e8f0 !important;
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
        font-family: 'Product Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: #2d3748 !important;
        border-right: none !important;
    }
    
    .css-1cypcdb {
        background: #2d3748 !important;
    }
    
    /* Sidebar text color fix */
    .sidebar [data-testid="stMarkdownContainer"] p {
        color: white !important;
        font-family: 'Product Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    .sidebar [data-testid="stMarkdownContainer"] h3 {
        color: white !important;
        font-family: 'Product Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* Radio buttons in sidebar */
    .stRadio > div {
        background: transparent !important;
    }
    
    .stRadio > div > div > div > div {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        border-radius: 8px !important;
        padding: 12px 16px !important;
        margin: 4px 0 !important;
        color: white !important;
        font-family: 'Product Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    .stRadio > div > div > div > div:hover {
        background: rgba(255, 255, 255, 0.2) !important;
    }
    
    .stRadio > div > div > div > div > label {
        color: white !important;
        font-family: 'Product Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* Form Elements */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        font-size: 14px !important;
        font-family: 'Product Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        font-size: 14px !important;
        padding: 12px 16px !important;
        font-family: 'Product Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    .stTextArea > div > div > textarea {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        font-size: 14px !important;
        padding: 12px 16px !important;
        font-family: 'Product Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* Label text fix */
    .stSelectbox > label, .stTextInput > label, .stTextArea > label {
        color: #ffffff !important;
        font-family: 'Product Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-weight: 500 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        border: none !important;
        border-radius: 12px !important;
        color: white !important;
        font-weight: 500 !important;
        font-size: 14px !important;
        padding: 12px 24px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1) !important;
        font-family: 'Product Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px 0 rgba(99, 102, 241, 0.4) !important;
    }
    
    /* Metrics */
    .stMetric > div {
        background: rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .stMetric > div > div > div {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-family: 'Product Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    .stMetric label {
        color: #e2e8f0 !important;
        font-family: 'Product Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* Plotly Charts */
    .js-plotly-plot {
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.3) !important;
    }
    
    /* File Uploader */
    .stFileUploader > div > div {
        background: rgba(255, 255, 255, 0.1) !important;
        border: 2px dashed rgba(255, 255, 255, 0.3) !important;
        border-radius: 16px !important;
        padding: 2rem !important;
    }
    
    /* Alerts */
    .stAlert {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        background: rgba(255, 255, 255, 0.1) !important;
    }
    
    .stSuccess {
        background: #f0fdf4 !important;
        border-color: #22c55e !important;
        color: #15803d !important;
    }
    
    .stError {
        background: #fef2f2 !important;
        border-color: #ef4444 !important;
        color: #dc2626 !important;
    }
    
    .stInfo {
        background: #eff6ff !important;
        border-color: #3b82f6 !important;
        color: #1d4ed8 !important;
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        padding-top: 2rem;
    }
    
    .sidebar-section {
        background: rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 1rem;
        backdrop-filter: blur(10px);
    }
    
    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: white !important;
        margin-bottom: 1rem;
        font-family: 'Product Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    /* Sliders */
    .stSlider > div > div > div > div {
        background: #e2e8f0 !important;
        border-radius: 8px !important;
    }
    
    .stSlider > div > div > div > div > div {
        background: #6366f1 !important;
        border-radius: 50% !important;
        box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3) !important;
    }
    
    .stSlider > label {
        color: #ffffff !important;
        font-family: 'Product Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
        font-weight: 500 !important;
    }
    
    /* Overview cards */
    .overview-card {
        background: rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.3);
        backdrop-filter: blur(10px);
    }
    
    .overview-card h3 {
        color: #ffffff !important;
        font-weight: 600;
        margin-bottom: 1rem;
        font-family: 'Product Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    
    .overview-card p {
        color: #e2e8f0 !important;
        margin: 0;
        font-family: 'Product Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Dashboard header
    st.markdown("""
    <div class="dashboard-header">
        <h1 class="dashboard-title">Timeline Visualizer</h1>
        <p class="dashboard-subtitle">Create beautiful, interactive timeline visualizations with AI-powered insights</p>
    </div>
    """, unsafe_allow_html=True)

    converter = SimpleTimelineConverter()
    visualizer = TimelineVisualizer()
    
    # Sidebar for controls
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown('<h3 class="sidebar-title">Navigation</h3>', unsafe_allow_html=True)
        
        # Navigation menu
        page = st.radio(
            "Select a function:",
            ["Dashboard", "AI Generate", "Upload Data", "Manual Input", "Visualize"],
            key="navigation"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick stats if data exists
        if st.session_state.timeline_data is not None and not st.session_state.timeline_data.empty:
            df = st.session_state.timeline_data
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.markdown('<h3 class="sidebar-title">Current Dataset</h3>', unsafe_allow_html=True)
            st.metric("Total Events", len(df))
            st.metric("Date Range", f"{df['year'].max() - df['year'].min()} years")
            st.metric("Avg Importance", f"{df['importance'].mean():.1f}/5")
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Main content area
    if page == "Dashboard":
        # Overview dashboard
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.session_state.timeline_data is not None:
                st.metric("Total Events", len(st.session_state.timeline_data))
            else:
                st.metric("Total Events", "0")
        
        with col2:
            st.metric("Manual Events", len(st.session_state.manual_events))
        
        with col3:
            if st.session_state.timeline_data is not None and len(st.session_state.timeline_data) > 0:
                st.metric("Years Covered", f"{st.session_state.timeline_data['year'].max() - st.session_state.timeline_data['year'].min()}")
            else:
                st.metric("Years Covered", "0")
        
        with col4:
            if st.session_state.timeline_data is not None and len(st.session_state.timeline_data) > 0:
                st.metric("Avg Importance", f"{st.session_state.timeline_data['importance'].mean():.1f}")
            else:
                st.metric("Avg Importance", "0")
        
        st.markdown("### Getting Started")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="overview-card">
                <h3>AI Generate</h3>
                <p>Use ChatGPT or Claude to generate timelines. Simply provide a topic and time period to get professionally formatted events.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="overview-card">
                <h3>Upload Data</h3>
                <p>Upload CSV files with timeline data or paste data from external sources. Convert existing timeline formats easily.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="overview-card">
                <h3>Manual Input</h3>
                <p>Add events one by one with full control over every detail. Perfect for creating custom timelines with precision.</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div class="overview-card">
                <h3>Visualize</h3>
                <p>Create interactive network graphs, traditional timelines, Gantt charts and theme rivers to visualize your data.</p>
            </div>
            """, unsafe_allow_html=True)
    
    elif page == "AI Generate":
        st.markdown("### AI-Powered Timeline Generation")
        
        # Configuration section
        with st.container():
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Configuration")
                provider = st.selectbox("AI Provider", ["ChatGPT", "Claude"], key="llm_provider_select")
                n_events = st.slider("Number of Events", 5, 20, 10, 1, key="llm_gen_n_events")
                api_key = st.text_input(f"{provider} API Key", type="password", 
                                      value=os.environ.get("OPENAI_API_KEY" if provider=="ChatGPT" else "ANTHROPIC_API_KEY", ""), 
                                      key="llm_api_key_input")
            
            with col2:
                st.markdown("#### Timeline Details")
        topic = st.text_input("Timeline Topic", placeholder="e.g., Space Race, World War II", key="llm_gen_topic")
        time_period = st.text_input("Time Period", placeholder="e.g., 1950 to 1980", key="llm_gen_time_period")
        
        # Generate button
        if st.button("Generate Timeline", type="primary", key="llm_gen_btn", use_container_width=True):
            if not api_key:
                st.error("Please enter your API key.")
            elif not topic:
                st.error("Please enter a timeline topic.")
            else:
                with st.spinner("AI is generating your timeline..."):
                    prompt = converter.get_llm_template(topic, time_period).replace("10-15", str(n_events))
                    llm_output = get_llm_timeline(prompt, provider, api_key)
                
                events = converter.parse_template_response(llm_output)
                if not events:
                    events = converter.smart_parse_llm_output(llm_output)
                
                if events:
                    df = converter.create_dataframe(events)
                    st.session_state.timeline_data = df
                    st.success(f"Successfully generated {len(events)} events!")
                    
                    # Show preview
                    st.markdown("### Generated Timeline Preview")
                    fig = visualizer._create_enhanced_timeline(df, 1.0, 5)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    with st.expander("View Raw Timeline Data"):
                        st.code(converter.to_standard_format(df), language="text")
                else:
                    st.error("Could not generate timeline. Please try again with different parameters.")

    elif page == "Upload Data":
        st.markdown("### Upload Timeline Data")
        
        upload_option = st.radio("Choose upload method:", ["Upload CSV File", "Paste Text Data"], horizontal=True)
        
        if upload_option == "Upload CSV File":
            uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], key="csv_uploader")
            if uploaded_file is not None:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.session_state.timeline_data = df
                    st.success(f"Successfully uploaded {len(df)} events!")
                    
                    # Show preview
                    st.markdown("### Data Preview")
                    st.dataframe(df.head(), use_container_width=True)
                except Exception as e:
                    st.error(f"Error reading CSV: {str(e)}")
        
        else:  # Paste data
            st.markdown("#### Paste Timeline Data")
        llm_response = st.text_area(
            "Paste your timeline data here:",
            height=200,
            placeholder="Paste timeline data from ChatGPT, Claude, or other sources...",
            key="llm_response_text_area"
        )
            
        if llm_response and st.button("Convert Data", type="primary", use_container_width=True):
            events = converter.parse_template_response(llm_response)
            if not events:
                events = converter.smart_parse_llm_output(llm_response)
                
            if events:
                df = converter.create_dataframe(events)
                st.session_state.timeline_data = df
                st.success(f"Successfully converted {len(events)} events!")
                
                # Show preview
                st.markdown("### Converted Timeline Preview")
                fig = visualizer._create_enhanced_timeline(df, 1.0, 5)
                st.plotly_chart(fig, use_container_width=True)
                
                with st.expander("View Converted Timeline"):
                    st.code(converter.to_standard_format(df), language="text")
        else:
            st.error("No events found. Please check your data format.")

    elif page == "Manual Input":
        st.markdown("### Manual Event Input")
        
        # Event input form
        with st.form("add_event_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### Event Details")
                date_input = st.date_input("Event Date", key="manual_date_input")
                title_input = st.text_input("Event Title", max_chars=50, key="manual_title_input")
                description_input = st.text_area("Description", max_chars=150, height=80, key="manual_description_text_area")
            
            with col2:
                st.markdown("#### Classification")
                importance_input = st.selectbox("Importance Level", [1, 2, 3, 4, 5], index=2, 
                                              help="1=Minor, 5=Major", key="manual_importance_select")
                subjects_input = st.text_input("Subjects", placeholder="comma, separated, values", key="manual_subjects_input")
            
            if st.form_submit_button("Add Event", type="primary", use_container_width=True):
                if title_input and description_input:
                    event = {
                        'date': pd.to_datetime(date_input),
                        'title': title_input,
                        'description': description_input,
                        'importance': importance_input,
                        'subjects': subjects_input
                    }
                    st.session_state.manual_events.append(event)
                    st.success("Event added successfully!")
                    st.rerun()
                else:
                    st.error("Please fill in all required fields.")
        
        # Display current events
        if st.session_state.manual_events:
            st.markdown("### Current Events")
            
            for i, event in enumerate(st.session_state.manual_events):
                with st.container():
                    col1, col2 = st.columns([4, 1])
                    
                    with col1:
                        subjects_text = f" | {event['subjects']}" if event.get('subjects') else ""
                        st.markdown(f"""
                        **{event['title']}**  
                        Date: {event['date'].strftime('%Y-%m-%d')} | Importance: {event['importance']}/5{subjects_text}  
                        {event['description']}
                        """)
                    
                    with col2:
                        if st.button("Delete", key=f"delete_event_{i}", help="Delete event"):
                            st.session_state.manual_events.pop(i)
                            st.success("Event deleted!")
                            st.rerun()
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Use for Visualization", key="use_manual_for_viz_btn", use_container_width=True):
                    df = converter.create_dataframe(st.session_state.manual_events)
                    st.session_state.timeline_data = df
                    st.success("Ready for visualization!")
            
            with col2:
                if st.button("Download Timeline", key="download_manual_timeline_btn", use_container_width=True):
                    df = converter.create_dataframe(st.session_state.manual_events)
                    standard_format = converter.to_standard_format(df)
                    st.download_button("Download", standard_format, "manual_timeline.txt", "text/plain")
            
            with col3:
                if st.button("Clear All", key="clear_manual_events_btn", use_container_width=True):
                    st.session_state.manual_events = []
                    st.rerun()
    
    elif page == "Visualize":
        st.markdown("### Timeline Visualization")
        
        if st.session_state.timeline_data is not None and not st.session_state.timeline_data.empty:
            df = st.session_state.timeline_data
            
            # Tile-based Controls
            st.markdown("### Visualization Controls")
            
            # Main control tabs
            control_tab1, control_tab2, control_tab3 = st.tabs([
                "Chart Type", "Settings", "Analysis"
            ])
            
            # Initialize parameters
            algorithm = "Force"
            proximity = "Cosine"
            relevance = "Importance"
            layout = "Force"
            node_size_range = (10, 50)
            edge_threshold = 0.1
            
            with control_tab1:  # Chart Type
                st.markdown("**Choose Visualization**")
                
                # Tile-based chart selection
                col1, col2 = st.columns(2)
            with col1:
                chart_type = st.selectbox(
                    "Chart Type",
                    ["Timeline", "Interactive Network", "Scatter Plot", "Gantt Chart", "Theme River"],
                    key="chart_type_selectbox"
                )
                    
                    # Additional controls based on chart type
                group_by = None
                if chart_type == "Theme River":
                        group_by = st.selectbox("Group By", ["Year", "Month", "Day"], key="theme_river_group_by")
            
            with col2:
                    st.markdown("**Dataset Info**")
                    st.info(f"Current Dataset: {len(df)} events")
                    st.write(f"Date Range: {df['year'].min()} - {df['year'].max()}")
            
            with control_tab2:  # Settings
                st.markdown("**Adjust Settings**")
                
                # Tile-based settings layout
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**General Settings**")
                    detail_level = st.slider("Detail Level", 1, 10, 5, 1, help="1=Overview, 10=Maximum Detail")
                    zoom_factor = st.slider("Zoom Factor", 0.5, 3.0, 1.0, 0.1, help="Adjust zoom level")
                
                with col2:
                    st.markdown("**Display Settings**")
                    connection_level = st.slider("Connection Level", 1, 5, 1, 1, help="1=Simple, 5=All connections")
                    size_multiplier = st.slider("Size Multiplier", 0.5, 3.0, 1.0, 0.1, help="Adjust element sizes")
            
            with col3:
                    st.markdown("**Network Settings**")
                    if chart_type == "Interactive Network":
                        algorithm = st.selectbox("Algorithm", ["Force", "Circular", "Random"], key="algorithm_select")
                        proximity = st.selectbox("Proximity", ["Cosine", "Pearson"], key="proximity_select")
                        
                        col1_inner, col2_inner = st.columns(2)
                        with col1_inner:
                            min_size = st.slider("Min Node Size", 5, 30, 10, key="min_size_slider")
                        with col2_inner:
                            max_size = st.slider("Max Node Size", 20, 80, 50, key="max_size_slider")
                        node_size_range = (min_size, max_size)
            
            with control_tab3:  # Analysis
                st.markdown("**Quick Analysis**")
                
                # Tile-based metrics layout
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Events", len(df))
                with col2:
                    st.metric("Date Range", f"{df['year'].max() - df['year'].min()} years")
                with col3:
                    st.metric("Avg Importance", f"{df['importance'].mean():.1f}/5")
                with col4:
                    st.metric("Most Common Year", df['year'].mode().iloc[0] if not df['year'].mode().empty else "N/A")
            
            # Selection controls in tile layout
            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**TOTAL: {len(df)} / SELECTED: 0**")
            with col2:
                if st.button("Select All", key="select_all_btn"):
                    st.info("All nodes selected")
            with col3:
                if st.button("Clear Selection", key="clear_selection_btn"):
                    st.info("Selection cleared")
            
            # Create visualization with zoom and detail controls
            if chart_type == "Interactive Network":
                fig = visualizer._create_enhanced_interactive_network(df, detail_level, connection_level)
            elif chart_type == "Scatter Plot":
                fig = visualizer._create_event_scatter_plot(df, detail_level)
            elif chart_type == "Gantt Chart":
                fig = visualizer._create_gantt_chart(df, detail_level)
            elif chart_type == "Theme River":
                fig = visualizer.create_timeline(df, chart_type, size_multiplier * zoom_factor, group_by, detail_level)
            else:
                fig = visualizer.create_timeline(df, chart_type, size_multiplier * zoom_factor, detail_level=detail_level)
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Comprehensive Legend Section
            st.markdown("### 📊 Visualization Legend")
            
            # Create legend based on chart type
            if chart_type == "Timeline":
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**🎯 Node Colors:**")
                    st.markdown("• **Blue**: Default importance color")
                    st.markdown("• **Subject Colors**: Based on event subjects")
                    st.markdown("• **Size**: Proportional to importance level")
                
                with col2:
                    st.markdown("**🔗 Connections:**")
                    st.markdown("• **Dotted Lines**: Connect sequential events")
                    st.markdown("• **Hover**: Shows full event details")
                    st.markdown("• **Detail Level**: Controls text visibility")
                
                with col3:
                    st.markdown("**⚙️ Controls:**")
                    st.markdown(f"• **Detail Level**: {detail_level}/10")
                    st.markdown(f"• **Size Multiplier**: {size_multiplier}x")
                    st.markdown(f"• **Zoom Factor**: {zoom_factor}x")
            
            elif chart_type == "Interactive Network":
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**🔴 Edge Types:**")
                    st.markdown("• **Red Lines**: Temporal connections")
                    st.markdown("• **Green Lines**: Subject-based connections")
                    st.markdown("• **Blue Lines**: Importance-based connections")
                
                with col2:
                    st.markdown("**🔵 Node Properties:**")
                    st.markdown("• **Size**: Importance + connection count")
                    st.markdown("• **Color**: Subject or importance based")
                    st.markdown("• **Hover**: Event details + connections")
                
                with col3:
                    st.markdown("**⚙️ Network Settings:**")
                    st.markdown(f"• **Connection Level**: {connection_level}/5")
                    st.markdown(f"• **Detail Level**: {detail_level}/10")
                    st.markdown("• **Algorithm**: Force-directed layout")
            
            elif chart_type == "Scatter Plot":
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**🔵 Point Properties:**")
                    st.markdown("• **Size**: Importance level")
                    st.markdown("• **Color**: Subject or importance")
                    st.markdown("• **Position**: Year vs Importance")
                
                with col2:
                    st.markdown("**📊 Axes:**")
                    st.markdown("• **X-axis**: Years")
                    st.markdown("• **Y-axis**: Importance levels (1-5)")
                    st.markdown("• **Each Point**: Individual event")
                
                with col3:
                    st.markdown("**⚙️ Settings:**")
                    st.markdown(f"• **Detail Level**: {detail_level}/10")
                    st.markdown("• **Hover**: Event details")
                    st.markdown("• **Text Labels**: Event titles")
            
            elif chart_type == "Gantt Chart":
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**📅 Bar Properties:**")
                    st.markdown("• **Horizontal Bars**: Each event")
                    st.markdown("• **Color**: Subject or importance")
                    st.markdown("• **Position**: Chronological order")
                
                with col2:
                    st.markdown("**📊 Axes:**")
                    st.markdown("• **X-axis**: Date timeline")
                    st.markdown("• **Y-axis**: Event titles")
                    st.markdown("• **Bar Width**: 1 day duration")
                
                with col3:
                    st.markdown("**⚙️ Settings:**")
                    st.markdown(f"• **Detail Level**: {detail_level}/10")
                    st.markdown("• **Hover**: Event details")
                    st.markdown("• **Height**: Auto-adjusted")
            
            elif chart_type == "Theme River":
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**🌊 Stream Layers:**")
                    st.markdown("• **Each Layer**: Represents a subject")
                    st.markdown("• **Width**: Total importance over time")
                    st.markdown("• **Color**: Unique per subject")
                
                with col2:
                    st.markdown("**📈 Data:**")
                    st.markdown(f"• **Grouped By**: {group_by}")
                    st.markdown("• **Y-axis**: Cumulative importance")
                    st.markdown("• **X-axis**: Time periods")
                
                with col3:
                    st.markdown("**⚙️ Controls:**")
                    st.markdown(f"• **Detail Level**: {detail_level}/10")
                    st.markdown(f"• **Size Multiplier**: {size_multiplier}x")
                    st.markdown("• **Hover**: Subject + importance")
            
            # Dataset info and analysis
            st.markdown("### Dataset Analysis")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Events", len(df))
            with col2:
                st.metric("Date Range", f"{df['year'].max() - df['year'].min()} years")
            with col3:
                st.metric("Avg Importance", f"{df['importance'].mean():.1f}/5")
            with col4:
                if 'subjects' in df.columns:
                    all_subjects = []
                    for subjects_str in df['subjects'].fillna(''):
                        if subjects_str:
                            all_subjects.extend([s.strip() for s in str(subjects_str).split(',') if s.strip()])
                    st.metric("Unique Subjects", len(set(all_subjects)) if all_subjects else 0)
                else:
                    st.metric("Unique Subjects", "0")
            
            # Export options
            st.markdown("### Export Options")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                csv_data = df.to_csv(index=False)
                st.download_button("CSV", csv_data, "timeline_data.csv", "text/csv", use_container_width=True)
            
            with col2:
                standard_format = converter.to_standard_format(df)
                st.download_button("Standard Format", standard_format, "timeline_standard.txt", "text/plain", use_container_width=True)
            
            with col3:
                json_data = df.to_json(orient='records', date_format='iso')
                st.download_button("JSON", json_data, "timeline_data.json", "application/json", use_container_width=True)
            
            with col4:
                if 'fig' in locals():
                    html_str = fig.to_html(include_plotlyjs='cdn')
                    st.download_button("Chart HTML", html_str, "timeline_chart.html", "text/html", use_container_width=True)
            
            # Data table
            st.markdown("### Timeline Data")
            display_df = df.copy()
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
            columns_to_show = ['date', 'title', 'description', 'importance']
            if 'subjects' in display_df.columns:
                columns_to_show.append('subjects')
            st.dataframe(display_df[columns_to_show], use_container_width=True)
        
        else:
            st.info("No timeline data available. Please create a timeline using one of the input methods.")
            
            # Sample data
            st.markdown("### Try with Sample Data")
            if st.button("Load Space Race Timeline Sample", key="load_sample_btn", use_container_width=True):
                sample_events = [
                    {
                        'date': pd.to_datetime('1957-10-04'),
                        'title': 'Sputnik 1 Launched',
                        'description': 'First artificial satellite launched by Soviet Union',
                        'importance': 5,
                        'subjects': 'Space Race, Cold War'
                    },
                    {
                        'date': pd.to_datetime('1961-04-12'),
                        'title': 'Yuri Gagarin First Human in Space',
                        'description': 'Soviet cosmonaut completes first human orbital flight',
                        'importance': 5,
                        'subjects': 'Space Race, Human Spaceflight'
                    },
                    {
                        'date': pd.to_datetime('1969-07-20'),
                        'title': 'Apollo 11 Moon Landing',
                        'description': 'Neil Armstrong and Buzz Aldrin land on the Moon',
                        'importance': 5,
                        'subjects': 'Space Race, Moon Landing, Apollo Program'
                    },
                    {
                        'date': pd.to_datetime('1971-04-19'),
                        'title': 'Salyut 1 Space Station',
                        'description': 'First space station launched by Soviet Union',
                        'importance': 4,
                        'subjects': 'Space Race, Space Stations'
                    },
                    {
                        'date': pd.to_datetime('1975-07-17'),
                        'title': 'Apollo-Soyuz Test Project',
                        'description': 'Joint US-Soviet space mission marks end of Space Race',
                        'importance': 4,
                        'subjects': 'Space Race, International Cooperation'
                    }
                ]
                df_sample = converter.create_dataframe(sample_events)
                st.session_state.timeline_data = df_sample
                st.success("Sample Space Race timeline loaded! Explore the visualization above.")
                st.rerun()

if __name__ == "__main__":
    main()