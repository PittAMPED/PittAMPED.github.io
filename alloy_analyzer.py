#!/usr/bin/env python3
"""
Alloy Database Parser and Visualizer
Parses Obsidian-formatted alloy data and creates interactive visualizations
"""

import os
import re
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import argparse


@dataclass
class AlloyData:
    """Data structure for a single alloy sample"""
    sample_id: str
    folder_path: str
    composition: Dict[str, float]  # Element -> amount
    primary_crystallization: Optional[float] = None
    secondary_crystallization: Optional[float] = None
    num_pinholes: Optional[int] = None
    pinhole_area_percent: Optional[float] = None
    curie_temperature: Optional[float] = None
    
    def to_dict(self):
        return asdict(self)


class PropertyExtractor:
    """Base class for extracting properties from markdown files"""
    
    def extract(self, content: str, sample_id: str) -> Dict[str, Any]:
        """Extract properties from file content"""
        raise NotImplementedError


class AlloyExtractor(PropertyExtractor):
    """Extract composition data from *Alloy.md files"""
    
    def extract(self, content: str, sample_id: str) -> Dict[str, Any]:
        composition = {}
        
        # Look for chemical formula patterns like Co-Fe2.5Mn2Nb2...
        formula_patterns = [
            r'([A-Z][a-z]?)(\d*\.?\d*)',  # Element followed by optional number
            r'Chemical\s+Formula[:\s]+(.+)',  # Explicit formula field
            r'Composition[:\s]+(.+)',  # Composition field
        ]
        
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            # Try to find formula in the line
            for pattern in formula_patterns[1:]:  # Skip first pattern for now
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    formula_text = match.group(1)
                    composition = self._parse_formula(formula_text)
                    if composition:
                        return {'composition': composition}
            
            # If no explicit formula found, try parsing the whole line as formula
            if any(c.isupper() for c in line) and any(c.isdigit() for c in line):
                composition = self._parse_formula(line)
                if composition:
                    return {'composition': composition}
        
        return {'composition': composition}
    
    def _parse_formula(self, formula: str) -> Dict[str, float]:
        """Parse chemical formula into element: amount dictionary"""
        composition = {}
        
        # Remove common prefixes and clean up
        formula = re.sub(r'^(Chemical Formula|Composition)[:\s]*', '', formula, flags=re.IGNORECASE)
        formula = formula.strip()
        
        # Pattern to match element followed by optional number
        pattern = r'([A-Z][a-z]?)(\d*\.?\d*)'
        matches = re.findall(pattern, formula)
        
        for element, amount_str in matches:
            if amount_str:
                try:
                    amount = float(amount_str)
                except ValueError:
                    amount = 1.0
            else:
                amount = 1.0
            composition[element] = amount
            
        return composition


class DSCExtractor(PropertyExtractor):
    """Extract crystallization data from *DSC.md files"""
    
    def extract(self, content: str, sample_id: str) -> Dict[str, Any]:
        data = {}
        
        # Look for crystallization peaks
        lines = content.split('\n')
        for i, line in enumerate(lines):
            line = line.strip().lower()
            
            # Primary crystallization
            if 'primary' in line and ('crystallization' in line or 'peak' in line):
                temp = self._extract_temperature(line, lines[i:i+3])
                if temp:
                    data['primary_crystallization'] = temp
            
            # Secondary crystallization
            elif 'secondary' in line and ('crystallization' in line or 'peak' in line):
                temp = self._extract_temperature(line, lines[i:i+3])
                if temp:
                    data['secondary_crystallization'] = temp
        
        return data
    
    def _extract_temperature(self, line: str, context_lines: List[str]) -> Optional[float]:
        """Extract temperature value from line or nearby context"""
        # Look for temperature patterns in current line and next few lines
        temp_patterns = [
            r'(\d+\.?\d*)\s*°?c',
            r'(\d+\.?\d*)\s*k',
            r'peak[:\s]*(\d+\.?\d*)',
            r'temperature[:\s]*(\d+\.?\d*)',
        ]
        
        for context_line in context_lines:
            for pattern in temp_patterns:
                match = re.search(pattern, context_line.lower())
                if match:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        continue
        return None


class MechanicalExtractor(PropertyExtractor):
    """Extract pinhole data from *Mechanical.md files"""
    
    def extract(self, content: str, sample_id: str) -> Dict[str, Any]:
        data = {}
        
        for line in content.split('\n'):
            line = line.strip().lower()
            
            # Number of pinholes
            if 'number' in line and 'pinhole' in line:
                num = self._extract_number(line)
                if num is not None:
                    data['num_pinholes'] = int(num)
            
            # Percentage area of pinholes
            elif ('area' in line or '%' in line) and 'pinhole' in line:
                percent = self._extract_number(line)
                if percent is not None:
                    data['pinhole_area_percent'] = percent
        
        return data
    
    def _extract_number(self, line: str) -> Optional[float]:
        """Extract numeric value from line"""
        # Look for numbers, possibly with % or other units
        patterns = [
            r'(\d+\.?\d*)\s*%',
            r':\s*(\d+\.?\d*)',
            r'(\d+\.?\d*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, line)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        return None


class VSMExtractor(PropertyExtractor):
    """Extract Curie temperature from *VSM.md files"""
    
    def extract(self, content: str, sample_id: str) -> Dict[str, Any]:
        data = {}
        
        for line in content.split('\n'):
            line = line.strip().lower()
            
            if 'curie' in line and ('temp' in line or 'temperature' in line):
                temp = self._extract_temperature(line)
                if temp:
                    data['curie_temperature'] = temp
                    break
        
        return data
    
    def _extract_temperature(self, line: str) -> Optional[float]:
        """Extract temperature value from line"""
        temp_patterns = [
            r'(\d+\.?\d*)\s*°?c',
            r'(\d+\.?\d*)\s*k',
            r':\s*(\d+\.?\d*)',
            r'(\d+\.?\d*)'
        ]
        
        for pattern in temp_patterns:
            match = re.search(pattern, line)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    continue
        return None


class AlloyDatabaseParser:
    """Main parser for the alloy database"""
    
    def __init__(self, database_path: str):
        self.database_path = Path(database_path)
        self.extractors = {
            'Alloy.md': AlloyExtractor(),
            'DSC.md': DSCExtractor(),
            'Mechanical.md': MechanicalExtractor(),
            'VSM.md': VSMExtractor()
        }
    
    def parse_all_alloys(self) -> List[AlloyData]:
        """Parse all alloys in the database"""
        alloys = []
        
        # Walk through the lauren folder and find alloy directories
        lauren_path = self.database_path / 'lauren'
        if not lauren_path.exists():
            print(f"Warning: Lauren folder not found at {lauren_path}")
            return alloys
        
        # Find all HOTG folders
        for hotg_folder in lauren_path.glob('HOTG*'):
            if hotg_folder.is_dir():
                print(f"Processing {hotg_folder.name}...")
                alloys.extend(self._parse_hotg_folder(hotg_folder))
        
        print(f"Found {len(alloys)} alloy samples total")
        return alloys
    
    def _parse_hotg_folder(self, hotg_path: Path) -> List[AlloyData]:
        """Parse all alloys in a HOTG folder"""
        alloys = []
        
        # Find numbered directories (alloy samples)
        for sample_dir in hotg_path.iterdir():
            if sample_dir.is_dir() and sample_dir.name.isdigit():
                try:
                    alloy = self._parse_single_alloy(sample_dir)
                    if alloy:
                        alloys.append(alloy)
                        print(f"  Parsed sample {alloy.sample_id}")
                except Exception as e:
                    print(f"  Error parsing {sample_dir}: {e}")
        
        return alloys
    
    def _parse_single_alloy(self, sample_path: Path) -> Optional[AlloyData]:
        """Parse a single alloy sample directory"""
        sample_id = f"{sample_path.parent.name}_{sample_path.name}"
        
        # Initialize alloy data
        alloy = AlloyData(
            sample_id=sample_id,
            folder_path=str(sample_path),
            composition={}
        )
        
        # Process each expected file type
        for file_suffix, extractor in self.extractors.items():
            file_pattern = f"{sample_path.name}{file_suffix}"
            file_path = sample_path / file_pattern
            
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Extract data using appropriate extractor
                    extracted_data = extractor.extract(content, sample_id)
                    
                    # Update alloy object with extracted data
                    for key, value in extracted_data.items():
                        if hasattr(alloy, key):
                            setattr(alloy, key, value)
                    
                except Exception as e:
                    print(f"    Error reading {file_path}: {e}")
        
        return alloy if alloy.composition else None


class AlloyVisualizer:
    """Create interactive visualizations of alloy data"""
    
    def __init__(self, alloys: List[AlloyData]):
        self.alloys = alloys
        self.df = self._create_dataframe()
    
    def _create_dataframe(self) -> pd.DataFrame:
        """Convert alloy data to pandas DataFrame for easy plotting"""
        data_rows = []
        
        for alloy in self.alloys:
            row = {
                'sample_id': alloy.sample_id,
                'primary_crystallization': alloy.primary_crystallization,
                'secondary_crystallization': alloy.secondary_crystallization,
                'num_pinholes': alloy.num_pinholes,
                'pinhole_area_percent': alloy.pinhole_area_percent,
                'curie_temperature': alloy.curie_temperature
            }
            
            # Add composition elements as separate columns
            for element, amount in alloy.composition.items():
                row[f'{element}'] = amount
            
            data_rows.append(row)
        
        return pd.DataFrame(data_rows)
    
    def get_available_properties(self) -> List[str]:
        """Get list of properties available for plotting"""
        numeric_columns = self.df.select_dtypes(include=['number']).columns.tolist()
        return [col for col in numeric_columns if col != 'sample_id']
    
    def create_scatter_plot(self, x_prop: str, y_prop: str, color_by: str = None, 
                          filters: Dict[str, Any] = None) -> go.Figure:
        """Create interactive scatter plot"""
        df_filtered = self.df.copy()
        
        # Apply filters
        if filters:
            for prop, value in filters.items():
                if prop in df_filtered.columns:
                    if isinstance(value, (list, tuple)):
                        df_filtered = df_filtered[df_filtered[prop].between(value[0], value[1])]
                    else:
                        df_filtered = df_filtered[df_filtered[prop] == value]
        
        # Remove rows where x or y properties are null
        df_filtered = df_filtered.dropna(subset=[x_prop, y_prop])
        
        if df_filtered.empty:
            fig = go.Figure()
            fig.add_annotation(
                text="No data available for selected properties and filters",
                xref="paper", yref="paper", x=0.5, y=0.5,
                showarrow=False
            )
            return fig
        
        # Create scatter plot
        if color_by and color_by in df_filtered.columns:
            fig = px.scatter(
                df_filtered,
                x=x_prop,
                y=y_prop,
                color=color_by,
                hover_data=['sample_id'],
                title=f'{y_prop} vs {x_prop}',
                labels={x_prop: x_prop.replace('_', ' ').title(),
                       y_prop: y_prop.replace('_', ' ').title()}
            )
        else:
            fig = px.scatter(
                df_filtered,
                x=x_prop,
                y=y_prop,
                hover_data=['sample_id'],
                title=f'{y_prop} vs {x_prop}',
                labels={x_prop: x_prop.replace('_', ' ').title(),
                       y_prop: y_prop.replace('_', ' ').title()}
            )
        
        fig.update_layout(height=600)
        return fig
    
    def create_composition_plot(self, y_prop: str, elements: List[str] = None) -> go.Figure:
        """Create plot showing how composition affects a property"""
        if not elements:
            # Find common elements
            element_cols = [col for col in self.df.columns 
                          if len(col) <= 2 and col[0].isupper() and col != 'sample_id']
            elements = element_cols[:5]  # Show top 5 elements
        
        df_filtered = self.df.dropna(subset=[y_prop])
        
        fig = make_subplots(rows=1, cols=len(elements), 
                           subplot_titles=elements,
                           shared_yaxes=True)
        
        for i, element in enumerate(elements, 1):
            if element in df_filtered.columns:
                element_data = df_filtered.dropna(subset=[element])
                fig.add_trace(
                    go.Scatter(
                        x=element_data[element],
                        y=element_data[y_prop],
                        mode='markers',
                        name=element,
                        text=element_data['sample_id'],
                        showlegend=(i == 1)
                    ),
                    row=1, col=i
                )
        
        fig.update_layout(
            title=f'{y_prop.replace("_", " ").title()} vs Element Composition',
            height=500
        )
        
        return fig


# Streamlit Interface
def create_streamlit_app():
    """Create Streamlit web interface"""
    st.set_page_config(page_title="Alloy Database Analyzer", layout="wide")
    
    st.title("🔬 Alloy Database Analyzer")
    st.markdown("Interactive analysis of alloy properties and composition")
    
    # Sidebar for database path and loading
    st.sidebar.header("Database Configuration")
    
    # Database path input
    default_path = st.sidebar.text_input(
        "Database Path", 
        value="./material_database",
        help="Path to the material_database folder"
    )
    
    # Load data button
    if st.sidebar.button("Load Database"):
        with st.spinner("Parsing alloy database..."):
            try:
                parser = AlloyDatabaseParser(default_path)
                alloys = parser.parse_all_alloys()
                
                if alloys:
                    st.session_state.alloys = alloys
                    st.session_state.visualizer = AlloyVisualizer(alloys)
                    st.success(f"Loaded {len(alloys)} alloy samples!")
                else:
                    st.error("No alloys found. Check the database path.")
                    
            except Exception as e:
                st.error(f"Error loading database: {e}")
    
    # Main interface (only show if data is loaded)
    if 'visualizer' in st.session_state:
        visualizer = st.session_state.visualizer
        
        # Display basic statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", len(visualizer.alloys))
        with col2:
            st.metric("Available Properties", len(visualizer.get_available_properties()))
        with col3:
            elements = [col for col in visualizer.df.columns 
                       if len(col) <= 2 and col[0].isupper()]
            st.metric("Elements Found", len(elements))
        
        # Property selection
        st.header("📊 Create Visualization")
        
        col1, col2 = st.columns(2)
        with col1:
            x_property = st.selectbox("X-axis Property", visualizer.get_available_properties())
        with col2:
            y_property = st.selectbox("Y-axis Property", 
                                    [p for p in visualizer.get_available_properties() 
                                     if p != x_property])
        
        # Optional color coding
        color_property = st.selectbox("Color by (optional)", 
                                    ["None"] + visualizer.get_available_properties())
        color_by = None if color_property == "None" else color_property
        
        # Filters
        st.subheader("🔍 Filters")
        filters = {}
        
        # Create filter controls for numeric properties
        filter_cols = st.columns(3)
        available_props = visualizer.get_available_properties()
        
        for i, prop in enumerate(available_props[:6]):  # Show first 6 properties
            with filter_cols[i % 3]:
                values = visualizer.df[prop].dropna()
                if len(values) > 0:
                    min_val, max_val = float(values.min()), float(values.max())
                    if min_val != max_val:
                        range_val = st.slider(
                            f"{prop.replace('_', ' ').title()}",
                            min_val, max_val, (min_val, max_val),
                            key=f"filter_{prop}"
                        )
                        if range_val != (min_val, max_val):
                            filters[prop] = range_val
        
        # Generate plot
        if st.button("Generate Plot"):
            with st.spinner("Creating visualization..."):
                fig = visualizer.create_scatter_plot(x_property, y_property, color_by, filters)
                st.plotly_chart(fig, use_container_width=True)
                
                # Option to download plot
                html_str = fig.to_html(include_plotlyjs='cdn')
                st.download_button(
                    label="Download as HTML",
                    data=html_str,
                    file_name=f"alloy_plot_{x_property}_vs_{y_property}.html",
                    mime="text/html"
                )
        
        # Data table
        st.header("📋 Data Table")
        if st.checkbox("Show raw data"):
            st.dataframe(visualizer.df, use_container_width=True)
    
    else:
        st.info("👆 Please load a database using the sidebar to begin analysis.")


# Command Line Interface
def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="Alloy Database Analyzer")
    parser.add_argument("database_path", help="Path to material database")
    parser.add_argument("--mode", choices=["cli", "streamlit"], default="cli",
                       help="Run mode: cli for command line, streamlit for web interface")
    parser.add_argument("--output", help="Output file path for generated plots")
    
    args = parser.parse_args()
    
    if args.mode == "streamlit":
        # Run Streamlit app
        os.system(f"streamlit run {__file__}")
    else:
        # Run CLI analysis
        print("🔬 Alloy Database Analyzer")
        print(f"Loading database from: {args.database_path}")
        
        # Parse database
        parser_obj = AlloyDatabaseParser(args.database_path)
        alloys = parser_obj.parse_all_alloys()
        
        if not alloys:
            print("❌ No alloys found!")
            return
        
        print(f"✅ Loaded {len(alloys)} alloy samples")
        
        # Create visualizer
        visualizer = AlloyVisualizer(alloys)
        
        # Show available properties
        properties = visualizer.get_available_properties()
        print(f"\n📊 Available properties: {', '.join(properties)}")
        
        # Interactive property selection
        print("\nSelect properties to plot:")
        x_prop = input(f"X-axis property ({'/'.join(properties[:3])}/...): ")
        y_prop = input(f"Y-axis property ({'/'.join(properties[:3])}/...): ")
        
        if x_prop in properties and y_prop in properties:
            # Create plot
            fig = visualizer.create_scatter_plot(x_prop, y_prop)
            
            # Save or show plot
            if args.output:
                fig.write_html(args.output)
                print(f"📁 Plot saved to: {args.output}")
            else:
                fig.show()
        else:
            print("❌ Invalid property names!")


if __name__ == "__main__":
    # Check if running in Streamlit
    try:
        import streamlit as st
        # If we can import streamlit and we're in a streamlit context
        if hasattr(st, '_is_running_with_streamlit'):
            create_streamlit_app()
        else:
            main()
    except ImportError:
        main()