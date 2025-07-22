import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
import re
import networkx as nx
from typing import List, Dict, Tuple

# Configure page
st.set_page_config(
    page_title="Timeline Converter & Visualizer",
    page_icon="📊",
    layout="wide"
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
            1: '#CCCCCC', 2: '#FFE4B5', 3: '#87CEEB', 4: '#98FB98', 5: '#FFB6C1'
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
                    # Validate date format
                    if re.match(r'\d{4}-\d{2}-\d{2}', date_str):
                        parsed_date = pd.to_datetime(date_str)
                    else:
                        # Try to extract year and create date
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
    """Simple timeline visualizer"""
    
    def __init__(self):
        self.importance_colors = {
            1: '#CCCCCC', 2: '#FFE4B5', 3: '#87CEEB', 4: '#98FB98', 5: '#FFB6C1'
        }
    
    def create_timeline(self, df: pd.DataFrame, chart_type: str = "Timeline", size_multiplier: float = 1.0, group_by: str = None, sankey_mode: str = None, hierarchy: str = None) -> go.Figure:
        """Create timeline visualization"""
        if df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No data to display", x=0.5, y=0.5, xref="paper", yref="paper", font=dict(size=16))
            return fig
        if chart_type == "Timeline":
            return self._create_enhanced_timeline(df, size_multiplier)
        elif chart_type == "Network":
            return self._create_network_graph(df, size_multiplier)
        elif chart_type == "Theme River":
            return self._create_theme_river(df, group_by)
        elif chart_type == "Gantt":
            return self._create_gantt_chart(df)
        elif chart_type == "Event Heatmap":
            return self._create_event_heatmap(df, group_by)
        elif chart_type == "Sankey Diagram":
            return self._create_sankey_diagram(df, sankey_mode)
        elif chart_type == "Sunburst":
            return self._create_sunburst(df, hierarchy)
        elif chart_type == "Treemap":
            return self._create_treemap(df, hierarchy)
        elif chart_type == "Correlation Matrix":
            return self._create_correlation_matrix(df)
        else:
            return self._create_enhanced_timeline(df, size_multiplier)

    def _create_enhanced_timeline(self, df: pd.DataFrame, size_multiplier: float) -> go.Figure:
        """Enhanced timeline with smart positioning"""
        fig = go.Figure()
        
        # Calculate positions to avoid overlap
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
        
        # Assign subject colors if available
        subject_colors = None
        if 'subjects' in df.columns:
            all_subjects = set()
            for s in df['subjects'].fillna(""):
                all_subjects.update([x.strip() for x in str(s).split(',') if x.strip()])
            palette = px.colors.qualitative.Plotly + px.colors.qualitative.Pastel
            subject_colors = {s: palette[i % len(palette)] for i, s in enumerate(sorted(all_subjects))}
        
        for i, (_, row) in enumerate(df.iterrows()):
            color = self.importance_colors.get(row['importance'], '#45B7D1')
            # Use subject color if available
            if subject_colors and 'subjects' in row and row['subjects']:
                first_subject = str(row['subjects']).split(',')[0].strip()
                if first_subject in subject_colors:
                    color = subject_colors[first_subject]
            size = max(15, row['importance'] * 10 * size_multiplier)
            
            hover_text = f"<b>{row['title']}</b><br>"
            hover_text += f"Date: {row['date'].strftime('%B %d, %Y')}<br>"
            hover_text += f"Description: {row['description']}<br>"
            hover_text += f"Importance: {row['importance']}/5"
            if subject_colors and 'subjects' in row and row['subjects']:
                hover_text += f"<br>Subjects: {row['subjects']}"
            
            fig.add_trace(go.Scatter(
                x=[row['date']],
                y=[y_positions[i]],
                mode='markers+text',
                marker=dict(size=size, color=color, line=dict(width=2, color='white')),
                text=row['title'][:40],
                textfont=dict(size=16, color='#fff'),
                textposition="top center" if i % 2 == 0 else "bottom center",
                hovertemplate=hover_text + "<extra></extra>",
                showlegend=False,
                name=row['title']
            ))
        
            # Add connection lines
            if i > 0:
                fig.add_trace(go.Scatter(
                    x=[df.iloc[i-1]['date'], row['date']],
                    y=[y_positions[i-1], y_positions[i]],
                    mode='lines',
                    line=dict(width=1.5, color='rgba(128, 128, 128, 0.4)', dash='dot'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        fig.update_layout(
            title=dict(text="Timeline Visualization", font=dict(size=24, color='#fff')),
            xaxis_title="Date",
            yaxis_title="Events",
            height=700,
            hovermode='closest',
            plot_bgcolor='#222',
            paper_bgcolor='#222',
            font=dict(color='#fff', size=16),
        )
        
        return fig
    
    def _create_network_graph(self, df: pd.DataFrame, size_multiplier: float) -> go.Figure:
        """Network graph showing event connections (less cluttered)"""
        if len(df) < 2:
            fig = go.Figure()
            fig.add_annotation(text="Need at least 2 events for network view", 
                             x=0.5, y=0.5, xref="paper", yref="paper")
            return fig
        G = nx.Graph()
        # Add nodes
        for i, row in df.iterrows():
            G.add_node(i, **row.to_dict())
        # Add edges: only between temporally adjacent events and those sharing a subject
        for i in range(len(df)):
            # Connect to next event (temporal adjacency)
            if i < len(df) - 1:
                G.add_edge(i, i+1)
            # Connect to other events with shared subject
            if 'subjects' in df.columns and df.iloc[i]['subjects']:
                subjects_i = set([s.strip() for s in str(df.iloc[i]['subjects']).split(',') if s.strip()])
                for j in range(i+1, len(df)):
                    if 'subjects' in df.columns and df.iloc[j]['subjects']:
                        subjects_j = set([s.strip() for s in str(df.iloc[j]['subjects']).split(',') if s.strip()])
                        if subjects_i & subjects_j:
                            G.add_edge(i, j)
        # Create layout (spread out more)
        pos = nx.spring_layout(G, k=2.5, iterations=100)
        # Edge traces
        edge_x, edge_y = [], []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'),
                               hoverinfo='none', mode='lines')
        # Node traces
        node_x, node_y, node_text, node_hover, node_sizes, node_colors = [], [], [], [], [], []
        for node in G.nodes():
            x, y = pos[node]
            row = df.iloc[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(row['title'][:20])
            node_hover.append(f"<b>{row['title']}</b><br>Date: {row['date'].strftime('%Y-%m-%d')}<br>Importance: {row['importance']}" + (f"<br>Subjects: {row['subjects']}" if 'subjects' in row and row['subjects'] else ""))
            node_sizes.append(22 + row['importance'] * 10 * size_multiplier)
            node_colors.append(row['importance'])
        node_trace = go.Scatter(
            x=node_x, y=node_y, mode='markers+text', text=node_text,
            marker=dict(size=node_sizes, color=node_colors, colorscale='Viridis',
                       colorbar=dict(title="Importance"), line=dict(width=2, color='white')),
            hovertemplate="%{hovertext}<extra></extra>", hovertext=node_hover
        )
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(title=dict(text='Network Graph', font=dict(size=24, color='#fff')), showlegend=False, height=700,
                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                         plot_bgcolor='#222', paper_bgcolor='#222', font=dict(color='#fff', size=16))
        return fig
    
    def _create_zoomable_timeline(self, df: pd.DataFrame, size_multiplier: float) -> go.Figure:
        """Zoomable timeline with range selector"""
        fig = go.Figure()
        
        for _, row in df.iterrows():
            color = self.importance_colors.get(row['importance'], '#45B7D1')
            size = max(15, row['importance'] * 12 * size_multiplier)
            
        fig.add_trace(go.Scatter(
                x=[row['date']],
                y=[row['importance']],
                mode='markers+text',
                marker=dict(size=size, color=color, line=dict(width=2, color='white')),
                text=row['title'][:30],
                textposition="top center",
                hovertemplate=f"<b>{row['title']}</b><br>Date: {row['date'].strftime('%Y-%m-%d')}<br>Importance: {row['importance']}/5<extra></extra>",
                showlegend=False
            ))
        
        fig.update_layout(
            title="Zoomable Timeline",
            xaxis_title="Date",
            yaxis_title="Importance (1-5)",
            height=700,
            xaxis=dict(
                rangeslider=dict(visible=True),
                type="date",
                rangeselector=dict(
                    buttons=[
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=5, label="5Y", step="year", stepmode="backward"),
                        dict(step="all", label="All")
                    ]
                )
            )
        )
        
        return fig
    
    def _create_gantt_chart(self, df: pd.DataFrame) -> go.Figure:
        """Gantt-style timeline (horizontal bars, one per event)"""
        # Assign subject colors if available
        bar_colors = []
        subject_colors = None
        if 'subjects' in df.columns:
            all_subjects = set()
            for s in df['subjects'].fillna(""):
                all_subjects.update([x.strip() for x in str(s).split(',') if x.strip()])
            palette = px.colors.qualitative.Plotly + px.colors.qualitative.Pastel
            subject_colors = {s: palette[i % len(palette)] for i, s in enumerate(sorted(all_subjects))}
        for _, row in df.iterrows():
            color = self.importance_colors.get(row['importance'], '#45B7D1')
            if subject_colors and 'subjects' in row and row['subjects']:
                first_subject = str(row['subjects']).split(',')[0].strip()
                if first_subject in subject_colors:
                    color = subject_colors[first_subject]
            bar_colors.append(color)
        # Each event is a horizontal bar
        fig = go.Figure()
        for i, (_, row) in enumerate(df.iterrows()):
            duration = timedelta(days=30 * row['importance'])
            end_date = row['date'] + duration
            fig.add_trace(go.Bar(
                x=[(end_date - row['date']).days],
                y=[row['title'][:40]],
                base=row['date'],
                orientation='h',
                marker=dict(color=bar_colors[i]),
                hovertemplate=f"<b>{row['title']}</b><br>Start: {row['date'].strftime('%Y-%m-%d')}<br>End: {end_date.strftime('%Y-%m-%d')}<br>Duration: {duration.days} days<extra></extra>",
                name=row['title']
            ))
        fig.update_layout(
            title=dict(text="Gantt Timeline", font=dict(size=24, color='#fff')),
            xaxis_title="Date",
            yaxis_title="Events",
            height=max(400, len(df) * 60),
            plot_bgcolor='#222',
            paper_bgcolor='#222',
            font=dict(color='#fff', size=16),
            barmode='stack',
            yaxis=dict(autorange="reversed"),
        )
        return fig

    def _create_theme_river(self, df: pd.DataFrame, group_by: str = None) -> go.Figure:
        """Theme River (streamgraph) visualization by subject and time granularity"""
        # Check for 'subjects' column
        if 'subjects' not in df.columns:
            fig = go.Figure()
            fig.add_annotation(text="No subject data for Theme River", x=0.5, y=0.5, xref="paper", yref="paper", font=dict(size=16))
            return fig
        # Parse subjects into list
        df = df.copy()
        df['subjects_list'] = df['subjects'].fillna("").apply(lambda x: [s.strip() for s in str(x).split(',') if s.strip()])
        # Explode subjects
        df_exploded = df.explode('subjects_list')
        if df_exploded['subjects_list'].isna().all() or df_exploded['subjects_list'].eq('').all():
            fig = go.Figure()
            fig.add_annotation(text="No subject data for Theme River", x=0.5, y=0.5, xref="paper", yref="paper", font=dict(size=16))
            return fig
        # Determine grouping
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
        # Group by selected granularity and subject, sum importance
        grouped = df_exploded.groupby(['group', 'subjects_list'])['importance'].sum().reset_index()
        # Pivot for area chart
        pivot = grouped.pivot(index='group', columns='subjects_list', values='importance').fillna(0)
        # Sort columns for consistent color order
        pivot = pivot.sort_index(axis=1)
        # Build area traces
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
            title=dict(text="Theme River (Subject Streamgraph)", font=dict(size=24, color='#fff')),
            xaxis_title=x_title,
            yaxis_title="Total Importance",
            height=700,
            hovermode='x unified',
            plot_bgcolor='#222',
            paper_bgcolor='#222',
            font=dict(color='#fff', size=16),
            legend_title="Subject"
        )
        return fig

    def _create_event_heatmap(self, df: pd.DataFrame, group_by: str = "Year") -> go.Figure:
        """Event heatmap: x=time (year/month), y=subject, color=event count or importance sum"""
        if 'subjects' not in df.columns:
            fig = go.Figure()
            fig.add_annotation(text="No subject data for heatmap", x=0.5, y=0.5, xref="paper", yref="paper", font=dict(size=16))
            return fig
        df = df.copy()
        df['subjects_list'] = df['subjects'].fillna("").apply(lambda x: [s.strip() for s in str(x).split(',') if s.strip()])
        df_exploded = df.explode('subjects_list')
        if group_by == "Month":
            df_exploded['group'] = df_exploded['date'].dt.to_period('M').astype(str)
            x_title = "Month"
        else:
            df_exploded['group'] = df_exploded['date'].dt.year
            x_title = "Year"
        # Pivot: index=subject, columns=group, values=event count
        heatmap_data = df_exploded.groupby(['subjects_list', 'group']).size().unstack(fill_value=0)
        # Optionally, use sum of importance:
        # heatmap_data = df_exploded.groupby(['subjects_list', 'group'])['importance'].sum().unstack(fill_value=0)
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns.astype(str),
            y=heatmap_data.index,
            colorscale='YlOrRd',
            colorbar=dict(title="Event Count")
        ))
        fig.update_layout(
            title=dict(text="Event Heatmap (Subjects vs. Time)", font=dict(size=24, color='#fff')),
            xaxis_title=x_title,
            yaxis_title="Subject/Entity",
            height=700,
            plot_bgcolor='#222',
            paper_bgcolor='#222',
            font=dict(color='#fff', size=16),
        )
        return fig

    def _create_sankey_diagram(self, df: pd.DataFrame, sankey_mode: str = "Year->Subject") -> go.Figure:
        """Sankey diagram: flows between years and subjects, or subject-to-subject transitions"""
        df = df.copy()
        df['subjects_list'] = df['subjects'].fillna("").apply(lambda x: [s.strip() for s in str(x).split(',') if s.strip()])
        df_exploded = df.explode('subjects_list')
        if sankey_mode == "Year->Subject":
            df_exploded['year'] = df_exploded['date'].dt.year
            links = df_exploded.groupby(['year', 'subjects_list']).size().reset_index(name='count')
            sources = links['year'].astype(str)
            targets = links['subjects_list']
        else:  # Subject->Subject (consecutive events)
            df_exploded = df_exploded.sort_values('date')
            sources = df_exploded['subjects_list'][:-1].values
            targets = df_exploded['subjects_list'][1:].values
            links = pd.DataFrame({'source': sources, 'target': targets})
            links = links.groupby(['source', 'target']).size().reset_index(name='count')
            sources = links['source']
            targets = links['target']
        all_labels = pd.Index(sources).append(pd.Index(targets)).unique().tolist()
        source_idx = [all_labels.index(s) for s in sources]
        target_idx = [all_labels.index(t) for t in targets]
        fig = go.Figure(go.Sankey(
            node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=all_labels),
            link=dict(source=source_idx, target=target_idx, value=links['count'])
        ))
        fig.update_layout(title_text="Sankey Diagram (Flows)", font=dict(size=16, color='#fff'), plot_bgcolor='#222', paper_bgcolor='#222')
        return fig

    def _create_sunburst(self, df: pd.DataFrame, hierarchy: str = "Subject>Year>Importance") -> go.Figure:
        df = df.copy()
        df['subjects_list'] = df['subjects'].fillna("").apply(lambda x: [s.strip() for s in str(x).split(',') if s.strip()])
        df_exploded = df.explode('subjects_list')
        df_exploded['year'] = df_exploded['date'].dt.year
        if hierarchy == "Subject>Year>Importance":
            path = [df_exploded['subjects_list'], df_exploded['year'].astype(str), df_exploded['importance'].astype(str)]
        elif hierarchy == "Year>Subject>Importance":
            path = [df_exploded['year'].astype(str), df_exploded['subjects_list'], df_exploded['importance'].astype(str)]
        else:
            path = [df_exploded['subjects_list'], df_exploded['year'].astype(str)]
        fig = px.sunburst(df_exploded, path=path, values=None, color='importance', color_continuous_scale='RdBu',
                          title="Sunburst: Hierarchical Event Breakdown")
        fig.update_layout(plot_bgcolor='#222', paper_bgcolor='#222', font=dict(color='#fff', size=16))
        return fig

    def _create_treemap(self, df: pd.DataFrame, hierarchy: str = "Subject>Year>Importance") -> go.Figure:
        df = df.copy()
        df['subjects_list'] = df['subjects'].fillna("").apply(lambda x: [s.strip() for s in str(x).split(',') if s.strip()])
        df_exploded = df.explode('subjects_list')
        df_exploded['year'] = df_exploded['date'].dt.year
        if hierarchy == "Subject>Year>Importance":
            path = [df_exploded['subjects_list'], df_exploded['year'].astype(str), df_exploded['importance'].astype(str)]
        elif hierarchy == "Year>Subject>Importance":
            path = [df_exploded['year'].astype(str), df_exploded['subjects_list'], df_exploded['importance'].astype(str)]
        else:
            path = [df_exploded['subjects_list'], df_exploded['year'].astype(str)]
        fig = px.treemap(df_exploded, path=path, values=None, color='importance', color_continuous_scale='RdBu',
                         title="Treemap: Hierarchical Event Breakdown")
        fig.update_layout(plot_bgcolor='#222', paper_bgcolor='#222', font=dict(color='#fff', size=16))
        return fig

    def _create_correlation_matrix(self, df: pd.DataFrame) -> go.Figure:
        df = df.copy()
        df['subjects_list'] = df['subjects'].fillna("").apply(lambda x: [s.strip() for s in str(x).split(',') if s.strip()])
        all_subjects = sorted(set(s for sublist in df['subjects_list'] for s in (sublist if isinstance(sublist, list) else [])))
        # Build co-occurrence matrix
        matrix = pd.DataFrame(0, index=all_subjects, columns=all_subjects)
        for subs in df['subjects_list']:
            if isinstance(subs, list):
                for i in range(len(subs)):
                    for j in range(i+1, len(subs)):
                        matrix.loc[subs[i], subs[j]] += 1
                        matrix.loc[subs[j], subs[i]] += 1
        fig = go.Figure(data=go.Heatmap(
            z=matrix.values,
            x=matrix.columns,
            y=matrix.index,
            colorscale='Blues',
            colorbar=dict(title="Co-occurrence")
        ))
        fig.update_layout(
            title=dict(text="Event Correlation Matrix (Subject Co-occurrence)", font=dict(size=24, color='#fff')),
            xaxis_title="Subject/Entity",
            yaxis_title="Subject/Entity",
            height=700,
            plot_bgcolor='#222',
            paper_bgcolor='#222',
            font=dict(color='#fff', size=16),
        )
        return fig

def main():
    """Main application with simple UI"""
    
    st.title("📊 Timeline Converter & Visualizer")
    st.markdown("**Convert LLM output to timeline format and create visualizations**")

    # Add default instruction block for LLM input
    st.markdown("""
    ### 📝 LLM Timeline Input Instructions
    Copy and paste this template into your LLM (ChatGPT, Claude, etc.) to generate timeline events in the correct format:
    """)
    st.code(
        '''Create a timeline in EXACT format: YYYY-MM-DD | Title | Description | Importance(1-5) | Subjects

Rules:
- One event per line
- Use exact date format YYYY-MM-DD
- Keep titles under 50 characters
- Keep descriptions under 150 characters
- Rate importance 1-5
- List one or more subjects (comma-separated) for each event (required for Theme River)
- No emojis, bullets, or formatting

Topic: [Your Topic]
Time Period: [Start] to [End]''',
        language="text"
    )
    
    converter = SimpleTimelineConverter()
    visualizer = TimelineVisualizer()
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Template Method", "✍️ Manual Input", "📊 Upload CSV", "📈 Visualize"])
    
    with tab1:
        st.subheader("🎯 Template-Guided LLM Conversion (Recommended)")
        st.markdown("**Best results: Get your LLM to output in the perfect format**")
        # Template generation
        col1, col2 = st.columns([1, 1])
        with col1:
            topic = st.text_input("Timeline Topic", placeholder="e.g., Space Race, World War II", key="timeline_topic_input")
            time_period = st.text_input("Time Period", placeholder="e.g., 1950 to 1980", key="timeline_time_period_input")
        with col2:
            if st.button("🎯 Generate LLM Template", type="primary"):
                template = converter.get_llm_template(topic, time_period)
                st.session_state.llm_template = template
        # Show template
        if 'llm_template' in st.session_state:
            st.markdown("### 📋 Copy this template to your LLM:")
            st.code(st.session_state.llm_template, language="text")
            st.download_button(
                "📥 Download Template",
                st.session_state.llm_template,
                "llm_timeline_template.txt",
                "text/plain"
            )
        # Process LLM response
        st.markdown("### 📤 Paste LLM Response:")
        llm_response = st.text_area(
            "Paste the LLM's formatted response here:",
            height=300,
            placeholder="Paste the timeline response from ChatGPT, Claude, etc...",
            key="llm_response_text_area"
        )
        if llm_response and st.button("🔄 Convert Response", type="primary"):
            events = converter.parse_template_response(llm_response)
            if events:
                df = converter.create_dataframe(events)
                st.session_state.timeline_data = df
                st.success(f"✅ Successfully converted {len(events)} events!")
                # Preview
                standard_format = converter.to_standard_format(df)
                st.code(standard_format, language="text")
                st.download_button(
                    "💾 Download Timeline",
                    standard_format,
                    "timeline.txt",
                    "text/plain"
                )
            else:
                st.error("❌ No events found. Make sure the LLM response follows the template format.")

    with tab2:
        st.subheader("✍️ Manual Input")
        st.markdown("**Most accurate: Enter events manually**")
        # Manual input form
        with st.form("add_event_form"):
            col1, col2 = st.columns(2)
            with col1:
                date_input = st.date_input("Event Date", key="manual_date_input")
                title_input = st.text_input("Event Title", max_chars=50, key="manual_title_input")
            with col2:
                importance_input = st.selectbox("Importance (1-5)", [1, 2, 3, 4, 5], index=2, key="manual_importance_select")
                description_input = st.text_area("Description", max_chars=150, height=80, key="manual_description_text_area")
            if st.form_submit_button("➕ Add Event"):
                event = {
                    'date': pd.to_datetime(date_input),
                    'title': title_input,
                    'description': description_input,
                    'importance': importance_input
                }
                st.session_state.manual_events.append(event)
                st.success("Event added!")
                st.rerun()
        # Show current events
        if st.session_state.manual_events:
            st.markdown("### 📋 Current Events:")
            for i, event in enumerate(st.session_state.manual_events):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.write(f"{i+1}. {event['date'].strftime('%Y-%m-%d')} | {event['title']} | {event['description']} | {event['importance']}")
                with col2:
                    if st.button("🗑️", key=f"delete_{i}"):
                        st.session_state.manual_events.pop(i)
                        st.rerun()
            
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📊 Use for Visualization", key="use_manual_for_viz_btn"):
                    df = converter.create_dataframe(st.session_state.manual_events)
                    st.session_state.timeline_data = df
                    st.success("Ready for visualization!")
            with col2:
                if st.button("💾 Download Timeline", key="download_manual_timeline_btn"):
                    df = converter.create_dataframe(st.session_state.manual_events)
                    standard_format = converter.to_standard_format(df)
                    st.download_button("Download", standard_format, "manual_timeline.txt", "text/plain")
            with col3:
                if st.button("🗑️ Clear All", key="clear_manual_events_btn"):
                    st.session_state.manual_events = []
                    st.rerun()
    
    with tab3:
        st.subheader("📊 CSV Upload")
        st.markdown("**Bulk import: Upload a CSV file**")
        
        # CSV template
        st.markdown("### 📥 CSV Template:")
        template_csv = "Date,Title,Description,Importance\n1969-07-20,Moon Landing,Apollo 11 lands on lunar surface,5\n1961-04-12,First Human in Space,Yuri Gagarin orbits Earth,5"
        
        st.code(template_csv, language="csv")
        st.download_button("📥 Download CSV Template", template_csv, "timeline_template.csv", "text/csv")
        
        # File upload
        uploaded_file = st.file_uploader("Upload CSV file", type=['csv'], key="csv_upload_file_uploader")
        if uploaded_file:
            try:
                df_upload = pd.read_csv(uploaded_file)
                required_columns = ['Date', 'Title', 'Description', 'Importance']
                if all(col in df_upload.columns for col in required_columns):
                    # Convert to timeline format
                    events = []
                    for _, row in df_upload.iterrows():
                        events.append({
                            'date': pd.to_datetime(row['Date']),
                            'title': str(row['Title'])[:50],
                            'description': str(row['Description'])[:150],
                            'importance': int(row['Importance'])
                        })
                    df = converter.create_dataframe(events)
                    st.session_state.timeline_data = df
                    st.success(f"✅ Successfully loaded {len(events)} events from CSV!")
                    # Preview
                    st.dataframe(df[['date', 'title', 'description', 'importance']], use_container_width=True)
                else:
                    st.error(f"❌ CSV must have columns: {', '.join(required_columns)}")
            except Exception as e:
                st.error(f"❌ Error reading CSV: {str(e)}")
    
    with tab4:
        st.subheader("📈 Timeline Visualization")
        
        if st.session_state.timeline_data is not None and not st.session_state.timeline_data.empty:
            df = st.session_state.timeline_data
            
            # Controls
            col1, col2, col3 = st.columns(3)
            
            with col1:
                chart_type = st.selectbox(
                    "Chart Type",
                    ["Timeline", "Network", "Theme River", "Gantt", "Event Heatmap", "Sankey Diagram", "Sunburst", "Treemap", "Correlation Matrix"],
                    key="chart_type_selectbox"
                )
                group_by = None
                sankey_mode = None
                hierarchy = None
                if chart_type == "Theme River":
                    group_by = st.selectbox(
                        "Group By (Theme River)",
                        ["Year", "Month", "Day"],
                        key="theme_river_group_by_selectbox"
                    )
                elif chart_type == "Event Heatmap":
                    group_by = st.selectbox(
                        "Group By (Heatmap)",
                        ["Year", "Month"],
                        key="heatmap_group_by_selectbox"
                    )
                elif chart_type == "Sankey Diagram":
                    sankey_mode = st.selectbox(
                        "Sankey Mode",
                        ["Year->Subject", "Subject->Subject (Consecutive)"],
                        key="sankey_mode_selectbox"
                    )
                elif chart_type in ["Sunburst", "Treemap"]:
                    hierarchy = st.selectbox(
                        "Hierarchy",
                        ["Subject>Year>Importance", "Year>Subject>Importance", "Subject>Year"],
                        key="hierarchy_selectbox"
                    )
            
            with col2:
                size_multiplier = st.slider("Size Multiplier", 0.5, 2.0, 1.0, 0.1, key="size_multiplier_slider")
            
            with col3:
                st.metric("Total Events", len(df))
                st.metric("Date Range", f"{df['year'].min()}-{df['year'].max()}")
            
            # Create visualization
            if chart_type == "Theme River":
                fig = visualizer.create_timeline(df, chart_type, size_multiplier, group_by)
            elif chart_type == "Event Heatmap":
                fig = visualizer.create_timeline(df, chart_type, size_multiplier, group_by)
            elif chart_type == "Sankey Diagram":
                fig = visualizer.create_timeline(df, chart_type, size_multiplier, group_by, sankey_mode)
            elif chart_type in ["Sunburst", "Treemap"]:
                fig = visualizer.create_timeline(df, chart_type, size_multiplier, group_by, sankey_mode, hierarchy)
            elif chart_type == "Correlation Matrix":
                fig = visualizer.create_timeline(df, chart_type, size_multiplier)
            else:
                fig = visualizer.create_timeline(df, chart_type, size_multiplier)
            
            # Visualization descriptions for non-technical users
            viz_descriptions = {
                "Timeline": "A simple timeline showing events as points over time. Useful for seeing the sequence and importance of events.",
                "Network": "A network graph showing how events are connected, either by time or shared subjects. Useful for exploring relationships between events.",
                "Theme River": "A streamgraph showing how the importance of different subjects changes over time. Best for seeing trends and shifts in focus.",
                "Gantt": "A Gantt chart showing the duration and overlap of events. Useful for visualizing timelines with durations.",
                "Event Heatmap": "A heatmap showing the number of events for each subject over time. Great for spotting periods of high activity for each subject.",
                "Sankey Diagram": "A Sankey diagram showing flows between years and subjects, or transitions between subjects. Useful for visualizing how focus shifts over time or between topics.",
                "Sunburst": "A sunburst chart showing a hierarchical breakdown of events by subject, year, and importance. Good for drilling down into the structure of your data.",
                "Treemap": "A treemap showing a hierarchical breakdown of events by subject, year, and importance. Useful for comparing the size and importance of different groups.",
                "Correlation Matrix": "A matrix showing how often subjects appear together in events. Useful for finding subjects that frequently co-occur."
            }
            st.markdown(f"**About this visualization:** {viz_descriptions.get(chart_type, '')}")
            st.plotly_chart(fig, use_container_width=True)
            
            # Analysis
            col1, col2 = st.columns(2)
            
            with col1:
                # Importance distribution
                importance_counts = df['importance'].value_counts().sort_index()
                fig_imp = px.bar(x=importance_counts.index, y=importance_counts.values,
                               title="Importance Distribution", labels={'x': 'Importance', 'y': 'Count'})
                st.plotly_chart(fig_imp, use_container_width=True)
            
            with col2:
                # Events per year
                if 'year' in df.columns:
                    yearly_counts = df['year'].value_counts().sort_index()
                    fig_year = px.line(x=yearly_counts.index, y=yearly_counts.values,
                                     title="Events Per Year", labels={'x': 'Year', 'y': 'Count'})
                    st.plotly_chart(fig_year, use_container_width=True)
            
            # Export options
            st.markdown("### 💾 Export Options")
            export_col1, export_col2, export_col3 = st.columns(3)
            
            with export_col1:
                csv_data = df.to_csv(index=False)
                st.download_button("📄 Download CSV", csv_data, "timeline_data.csv", "text/csv")
            
            with export_col2:
                standard_format = converter.to_standard_format(df)
                st.download_button("📝 Download Standard Format", standard_format, "timeline_standard.txt", "text/plain")
            
            with export_col3:
                json_data = df.to_json(orient='records', date_format='iso')
                st.download_button("📋 Download JSON", json_data, "timeline_data.json", "application/json")
            
            # Data table
            st.markdown("### 📋 Timeline Data")
            display_df = df.copy()
            display_df['date'] = display_df['date'].dt.strftime('%Y-%m-%d')
            st.dataframe(display_df[['date', 'title', 'description', 'importance']], use_container_width=True)
        
        else:
            st.info("📥 No timeline data available. Please create a timeline using one of the input methods above.")
            
    # Sidebar with quick stats
    with st.sidebar:
        st.markdown("### 📊 Quick Info")
        
        if st.session_state.timeline_data is not None:
            df = st.session_state.timeline_data
            st.metric("Events", len(df))
            st.metric("Avg Importance", f"{df['importance'].mean():.1f}/5")
            
            if len(df) > 0:
                date_range = f"{df['year'].min()} - {df['year'].max()}"
                st.metric("Time Span", date_range)
        
        st.markdown("### 🎯 How to Use")
        st.markdown("""
        1. **Template Method**: Generate LLM template, paste response
        2. **Manual Input**: Add events one by one
        3. **CSV Upload**: Bulk import from spreadsheet
        4. **Visualize**: Create interactive charts
        """)
        
        st.markdown("### 💡 Tips")
        st.markdown("""
        - Template method gives best results
        - Use precise dates (YYYY-MM-DD)
        - Rate importance 1-5 consistently
        - Keep titles/descriptions concise
        """)

if __name__ == "__main__":
    main()