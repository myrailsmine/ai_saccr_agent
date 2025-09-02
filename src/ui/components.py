# src/ui/components.py

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Any, Optional
import pandas as pd

class UIComponents:
    """Reusable UI components for the SA-CCR application"""
    
    def __init__(self):
        pass
    
    def render_metric_card(self, title: str, value: str, delta: Optional[str] = None, 
                          delta_color: str = "normal"):
        """Render a professional metric card"""
        
        delta_html = ""
        if delta:
            color = "green" if delta_color == "normal" else "red"
            delta_html = f'<div style="color: {color}; font-size: 0.8rem; margin-top: 0.25rem;">{delta}</div>'
        
        st.markdown(f"""
        <div class="metric-container">
            <div class="result-label">{title}</div>
            <div class="result-value">{value}</div>
            {delta_html}
        </div>
        """, unsafe_allow_html=True)
    
    def render_progress_bar(self, progress: float, message: str = ""):
        """Render enhanced progress bar"""
        
        progress_html = f"""
        <div class="progress-container">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="font-weight: 600;">Calculation Progress</span>
                <span>{progress:.1f}%</span>
            </div>
            <div style="background: #e5e7eb; border-radius: 4px; height: 8px;">
                <div style="background: #2563eb; height: 100%; border-radius: 4px; 
                           width: {progress}%; transition: width 0.3s ease;"></div>
            </div>
            {f'<div style="margin-top: 0.5rem; color: #6b7280; font-size: 0.875rem;">{message}</div>' if message else ''}
        </div>
        """
        
        st.markdown(progress_html, unsafe_allow_html=True)
    
    def render_calculation_step_visual(self, steps: List[Dict], current_step: int = 0):
        """Render visual representation of calculation steps"""
        
        fig = go.Figure()
        
        # Create step flow visualization
        x_positions = list(range(len(steps)))
        y_position = [0] * len(steps)
        
        # Color steps based on status
        colors = []
        for i, step in enumerate(steps):
            if i < current_step:
                colors.append('#10b981')  # Green for completed
            elif i == current_step:
                colors.append('#f59e0b')  # Orange for current
            else:
                colors.append('#e5e7eb')  # Gray for pending
        
        # Add step markers
        fig.add_trace(go.Scatter(
            x=x_positions,
            y=y_position,
            mode='markers',
            marker=dict(
                size=20,
                color=colors,
                line=dict(width=2, color='white')
            ),
            text=[f"Step {i+1}" for i in range(len(steps))],
            textposition="middle center",
            hovertext=[step.get('title', f'Step {i+1}') for i, step in enumerate(steps)],
            hoverinfo='text',
            showlegend=False
        ))
        
        # Add connecting lines
        fig.add_trace(go.Scatter(
            x=x_positions,
            y=y_position,
            mode='lines',
            line=dict(color='#d1d5db', width=2),
            showlegend=False,
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title="SA-CCR Calculation Progress",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-1, 1]),
            plot_bgcolor='white',
            height=150,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        
        return fig
    
    def render_data_table(self, data: pd.DataFrame, title: str = "", 
                         height: int = 400, use_container_width: bool = True):
        """Render a styled data table"""
        
        if title:
            st.markdown(f"#### {title}")
        
        st.dataframe(
            data, 
            height=height, 
            use_container_width=use_container_width
        )
    
    def render_key_value_pairs(self, data: Dict[str, Any], title: str = ""):
        """Render key-value pairs in a clean format"""
        
        if title:
            st.markdown(f"#### {title}")
        
        for key, value in data.items():
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown(f"**{key}:**")
            with col2:
                st.write(value)
    
    def render_risk_gauge(self, value: float, max_value: float, title: str, 
                         thresholds: Dict[str, float] = None):
        """Render a risk gauge chart"""
        
        if thresholds is None:
            thresholds = {'low': 0.3, 'medium': 0.7, 'high': 1.0}
        
        # Normalize value
        normalized_value = value / max_value if max_value > 0 else 0
        
        # Determine risk level
        if normalized_value <= thresholds['low']:
            color = '#10b981'  # Green
            risk_level = 'Low'
        elif normalized_value <= thresholds['medium']:
            color = '#f59e0b'  # Orange
            risk_level = 'Medium'
        else:
            color = '#ef4444'  # Red
            risk_level = 'High'
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=normalized_value * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"{title}<br><span style='font-size:0.8em;color:gray'>{risk_level} Risk</span>"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, thresholds['low']*100], 'color': "lightgray"},
                    {'range': [thresholds['low']*100, thresholds['medium']*100], 'color': "gray"},
                    {'range': [thresholds['medium']*100, 100], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
        
        return fig
    
    def render_comparison_chart(self, before: Dict[str, float], after: Dict[str, float], 
                               title: str = "Before vs After Comparison"):
        """Render a before/after comparison chart"""
        
        categories = list(before.keys())
        before_values = list(before.values())
        after_values = list(after.values())
        
        fig = go.Figure(data=[
            go.Bar(name='Before', x=categories, y=before_values, marker_color='#ef4444'),
            go.Bar(name='After', x=categories, y=after_values, marker_color='#10b981')
        ])
        
        fig.update_layout(
            title=title,
            barmode='group',
            xaxis_title="Metrics",
            yaxis_title="Values",
            height=400
        )
        
        return fig
    
    def render_alert(self, message: str, alert_type: str = "info"):
        """Render styled alert messages"""
        
        alert_styles = {
            'info': {'bg': '#dbeafe', 'border': '#93c5fd', 'text': '#1e40af'},
            'warning': {'bg': '#fef3c7', 'border': '#fcd34d', 'text': '#92400e'},
            'success': {'bg': '#d1fae5', 'border': '#6ee7b7', 'text': '#065f46'},
            'error': {'bg': '#fee2e2', 'border': '#fca5a5', 'text': '#991b1b'}
        }
        
        style = alert_styles.get(alert_type, alert_styles['info'])
        
        st.markdown(f"""
        <div style="
            background-color: {style['bg']}; 
            border: 1px solid {style['border']}; 
            color: {style['text']}; 
            padding: 1rem; 
            border-radius: 8px; 
            margin: 1rem 0;
        ">
            {message}
        </div>
        """, unsafe_allow_html=True)
    
    def render_step_timeline(self, steps: List[Dict], current_step: int = 0):
        """Render a timeline of calculation steps"""
        
        for i, step in enumerate(steps):
            # Determine step status
            if i < current_step:
                status = "completed"
                icon = "✅"
                color = "#10b981"
            elif i == current_step:
                status = "in_progress"
                icon = "🔄"
                color = "#f59e0b"
            else:
                status = "pending"
                icon = "⏳"
                color = "#9ca3af"
            
            # Render step
            st.markdown(f"""
            <div style="
                display: flex; 
                align-items: center; 
                padding: 0.5rem; 
                margin: 0.25rem 0;
                border-left: 3px solid {color};
                background-color: {'#f8fafc' if status == 'completed' else 'white'};
            ">
                <span style="margin-right: 0.5rem; font-size: 1.2em;">{icon}</span>
                <div>
                    <strong>Step {step.get('step', i+1)}: {step.get('title', f'Step {i+1}')}</strong>
                    <br>
                    <small style="color: #6b7280;">{step.get('description', '')}</small>
                </div>
            </div>
            """, unsafe_allow_html=True)
