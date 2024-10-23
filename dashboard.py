import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import base64
import pydeck as pdk
import plotly.graph_objects as go

import streamlit as st
import requests
import plotly.express as px
from datetime import datetime, timedelta
import functions

st.set_page_config(page_title="PG Dashboard", page_icon="🌟", layout="wide")

# Load the logo image
logo_path = "smalllogo.png"

# Read the image file and encode it to Base64
with open(logo_path, "rb") as image_file:
    logo_image = base64.b64encode(image_file.read()).decode()

# Custom CSS to style the header and position the logo
st.markdown(
    """
    <style>
    .header {
        background-color: crimson;
        padding: 10px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-radius: 5px;
        color: white;
    }
    .header-title {
        font-size: 30px;
        margin: 0;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the header with logo
st.markdown(
    f"""
    <div class="header">
        <div class="header-title">Atmospheric Potential Gradient measurements</div>
        <div class="header-logo">
            <img src="data:image/png;base64,{logo_image}" alt="Logo" style="width: 50px; height: 50px;">
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Location", "Plots"))

# Page 1 Content
if page == "Location":
    st.title("The location of measurements")

    # Coordinates for Magdeburg locations
    locations = pd.DataFrame({
        'latitude': [52.13985205227813, 52.13953099888456, 52.139399836160344,52.140145696650606],  # Magdeburg Center, University, City Center
        'longitude': [11.679347665213113, 11.67917223160897,11.676275982206214, 11.677236082514947],
        'name': ['Location N1', 'Location N2', 'EFM', 'Roof'],
    })

    # Define the layer for markers
    markers_layer = pdk.Layer(
        "ScatterplotLayer",
        locations,
        get_position="[longitude, latitude]",
        get_color="[200, 30, 0, 160]",  # Red color for markers
        get_radius=15,  # Size of the markers
        pickable=True,
    )

    # Load GeoJSON for world borders
    world_geojson_url = "https://raw.githubusercontent.com/datasets/geo-boundaries-world-110m/master/countries.geojson"
    world_borders = requests.get(world_geojson_url).json()

    # Define the GeoJsonLayer for country borders
    borders_layer = pdk.Layer(
        "GeoJsonLayer",
        world_borders,
        get_fill_color="[0, 0, 0, 0]",  # No fill
        get_line_color="[0, 0, 0]",  # Black border
        line_width_min_pixels=1,
        stroked=True,
        filled=False,
    )

    # Define the layer for city names
    text_layer = pdk.Layer(
        "TextLayer",
        locations,
        get_position="[longitude, latitude]",
        get_text="name",
        get_color="[255, 255, 255]",  # White text
        get_size=12,
        size_scale=1,  # Static size for text
        pickable=True,
    )

    # Define the Pydeck view
    view_state = pdk.ViewState(
        latitude=52.139399836160344,
        longitude=11.676275982206214,
        zoom=16,  # Zoom level
        pitch=0,  # Flat view
    )

    # Render the deck.gl map with satellite style
    r = pdk.Deck(
        map_style="mapbox://styles/mapbox/satellite-v9",  # Satellite style
        initial_view_state=view_state,
        layers=[borders_layer, markers_layer, text_layer],
        tooltip={"text": "Location: {name}"},
    )

    # Display the map in Streamlit
    st.pydeck_chart(r)

# Page 2 Content
elif page == "Plots":
    st.title("Interactive Plot - Electric Field Parallel Measurements")

    # Load the datasets
    elmagde = pd.read_csv('C:/Users/gayak/Desktop/FMExperiment/FM_Magdeburg/CS110_Electric_Field_Meter.csv', 
                          names=['id', 'timestamp', 'Efield', 'sensstat', 'curr-na','tempdeg', 'sensvolt','interRH'], header=None)
    elmagde = elmagde[['timestamp', 'Efield']]
    
    file_path = 'C:/Users/gayak/Desktop/FMExperiment/CR1000_Tab60sec.dat'
    
    try:
        exdf = pd.read_csv(file_path).reset_index()
        exdf.columns = exdf.iloc[0]
        exdf = exdf[3:].reset_index(drop=True)
        exdf = exdf[['TIMESTAMP','E_field_Avg','status']]
    except FileNotFoundError:
        st.error(f"File '{file_path}' not found.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
    
    # Process timestamps
    exdf['TIMESTAMP'] = pd.to_datetime(exdf['TIMESTAMP'])
    exdf['timestamp'] = exdf['TIMESTAMP'] + timedelta(hours=2)
    exdf['timestamp'] = pd.to_datetime(exdf['timestamp'])
    exdf['E_field_Avg'] = pd.to_numeric(exdf['E_field_Avg'], errors='coerce')

    # Combine the timestamps from both datasets into a single datetime column
    elmagde['timestamp'] = pd.to_datetime(elmagde['timestamp'])

    # Specific dates and times for filtering
    date_ranges = [
        {'Date': '14.08.2024', 'Start': '10:15', 'End': '17:00'},
        {'Date': '16.08.2024', 'Start': '17:00', 'End': '20:50'},
        {'Date': '17.08.2024', 'Start': '11:00', 'End': '16:00'},
        {'Date': '18.08.2024', 'Start': '15:00', 'End': '18:00'},
        {'Date': '19.08.2024', 'Start': '10:00', 'End': '20:48'},
        # Add all remaining entries in this format...
        {'Date': '16.10.2024', 'Start': '10:00', 'End': '18:30'}
    ]

    # Convert date strings to datetime for filtering
    for date_range in date_ranges:
        date_range['Start'] = datetime.strptime(f"{date_range['Date']} {date_range['Start']}", "%d.%m.%Y %H:%M")
        date_range['End'] = datetime.strptime(f"{date_range['Date']} {date_range['End']}", "%d.%m.%Y %H:%M")

    # Allow users to select a specific date range
    date_options = [f"{d['Date']} {d['Start'].strftime('%H:%M')} - {d['End'].strftime('%H:%M')}" for d in date_ranges]
    selected_option = st.sidebar.selectbox("Select a date and time range:", date_options)
    
    # Find the corresponding date range
    selected_date_range = next((d for d in date_ranges if f"{d['Date']} {d['Start'].strftime('%H:%M')} - {d['End'].strftime('%H:%M')}" == selected_option), None)
    
    if selected_date_range:
        start_time = selected_date_range['Start']
        end_time = selected_date_range['End']
        
        # Adjusted time range: one day before and one day after the selected range
        adjusted_start_time = start_time - timedelta(days=1)
        adjusted_end_time = end_time + timedelta(days=1)
        
        # Filter data based on the adjusted time range
        elmagde_filtered = elmagde[(elmagde['timestamp'] >= adjusted_start_time) & (elmagde['timestamp'] <= adjusted_end_time)]
        
        # Apply removenan function on the filtered data
        elmagde_filtered = functions.removenan(elmagde_filtered)
        
        # Now, filter back to the original selected range
        elmagde_filtered = elmagde_filtered[(elmagde_filtered['timestamp'] >= start_time) & (elmagde_filtered['timestamp'] <= end_time)]
        
        # Filter data for exdf within the original selected range (no additional day adjustment for this dataset)
        exdf_filtered = exdf[(exdf['timestamp'] >= start_time) & (exdf['timestamp'] <= end_time)]
        
        # Calculate the min and max values across both datasets for unified y-axis
        min_value = min(elmagde_filtered['Efield'].min(), exdf_filtered['E_field_Avg'].min())
        max_value = max(elmagde_filtered['Efield'].max(), exdf_filtered['E_field_Avg'].max())
        
        # Create an interactive plot with two y-axes
        fig = go.Figure()

        # Add traces for each dataset with markers
        fig.add_trace(go.Scatter(x=elmagde_filtered['timestamp'], y=elmagde_filtered['Efield'],
                                 mode='lines+markers', name='Original EFM',
                                 line=dict(color='black'),
                                 marker=dict(size=5)))  # Markers added
        
        fig.add_trace(go.Scatter(x=exdf_filtered['timestamp'], y=exdf_filtered['E_field_Avg'],
                                 mode='lines+markers', name='Experiment',
                                 line=dict(color='crimson'),
                                 marker=dict(size=5),  # Markers added
                                 yaxis='y2'))  # Assign to secondary y-axis
        
        # Update layout to include a secondary y-axis with uniform range
        fig.update_layout(
            title=f'Electric Field Data from {selected_date_range["Start"].strftime("%d.%m.%Y %H:%M")} to {selected_date_range["End"].strftime("%d.%m.%Y %H:%M")}',
            xaxis_title='Timestamp',
            yaxis=dict(
                title='Electric Field (V/m) - Dataset 1',
                range=[min_value, max_value]  # Unified y-axis range
            ),
            yaxis2=dict(
                title='E Field Avg (V/m) - Dataset 2',
                overlaying='y',
                side='right',
                range=[min_value, max_value]  # Unified y-axis range
            ),
            legend=dict(x=0.01, y=0.99),
            template='plotly_dark'
        )
        
        # Show the plot in Streamlit
        st.plotly_chart(fig)