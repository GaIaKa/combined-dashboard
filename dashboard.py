import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import base64
import pydeck as pdk
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

import streamlit as st
import requests
import plotly.express as px
from datetime import datetime, timedelta
import functions

st.set_page_config(page_title="PG Dashboard", page_icon="☁", layout="wide")

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
    {'Date': '14.08.2024', 'Start': '10:15', 'End': '17:00', 'Location':'N1'},
    {'Date': '16.08.2024', 'Start': '17:00', 'End': '20:50', 'Location':'N1'},
    {'Date': '17.08.2024', 'Start': '11:00', 'End': '16:00', 'Location':'N1'},
    {'Date': '18.08.2024', 'Start': '15:00', 'End': '18:00', 'Location':'N1'},
    {'Date': '19.08.2024', 'Start': '10:00', 'End': '20:48', 'Location':'N1'},
    {'Date': '20.08.2024', 'Start': '16:45', 'End': '20:40', 'Location':'N1'},
    {'Date': '22.08.2024', 'Start': '13:00', 'End': '20:12', 'Location':'N1'},
    {'Date': '24.08.2024', 'Start': '06:38', 'End': '20:05', 'Location':'N1'},
    {'Date': '26.08.2024', 'Start': '12:30', 'End': '19:40', 'Location':'N2'},
    {'Date': '28.08.2024', 'Start': '10:45', 'End': '19:45', 'Location':'N2'},
    {'Date': '29.08.2024', 'Start': '12:00', 'End': '20:00', 'Location':'N2'},
    {'Date': '01.09.2024', 'Start': '07:20', 'End': '19:55', 'Location':'N2'},
    {'Date': '02.09.2024', 'Start': '16:30', 'End': '00:00', 'Location':'N2'},
    {'Date': '03.09.2024', 'Start': '11:25', 'End': '20:05', 'Location':'N2'},
    {'Date': '05.09.2024', 'Start': '13:50', 'End': '16:40', 'Location':'Next toEFM'},
    {'Date': '06.09.2024', 'Start': '18:14', 'End': '19:10', 'Location':'Roof'},
    {'Date': '07.09.2024', 'Start': '20:50', 'End': '23:59','Location':'Roof'},
    {'Date': '08.09.2024', 'Start': '00:00', 'End': '23:59','Location':'Roof'},
    {'Date': '09.09.2024', 'Start': '00:00', 'End': '07:20','Location':'Roof'},
    {'Date': '16.09.2024', 'Start': '14:11', 'End': '23:59' ,'Location':'Next to EFM/Roof'},
    {'Date': '17.09.2024', 'Start': '00:00', 'End': '23:59','Location':'Roof'},
    {'Date': '18.09.2024', 'Start': '00:00', 'End': '13:18','Location':'Roof'},
    {'Date': '19.09.2024', 'Start': '08:30', 'End': '23:59','Location':'Roof'},
    {'Date': '20.09.2024', 'Start': '00:00', 'End': '19:00','Location':'Roof'},
    {'Date': '22.09.2024', 'Start': '07:12', 'End': '23:59','Location':'Roof'},
    {'Date': '23.09.2024', 'Start': '00:00', 'End': '17:50', 'Location':'Roof'},
    {'Date': '25.09.2024', 'Start': '14:55', 'End': '18:15','Location':'Bottom of the field'},
    {'Date': '28.09.2024', 'Start': '13:45', 'End': '19:13', 'Location':'N2'},
    {'Date': '29.09.2024', 'Start': '10:19', 'End': '18:58','Location':'N2'},
    {'Date': '04.10.2024', 'Start': '12:00', 'End': '14:50','Location':'N2'},
    {'Date': '06.10.2024', 'Start': '10:49', 'End': '17:57','Location':'N2'},
    {'Date': '11.10.2024', 'Start': '10:00', 'End': '18:10','Location':'N2'},
    {'Date': '12.10.2024', 'Start': '10:30', 'End': '17:57','Location':'N2'},
    {'Date': '16.10.2024', 'Start': '10:00', 'End': '18:30','Location':'N2'}
]

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ("Location", "Interactive Plot"))

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
elif page == "Interactive Plot":
    st.title("Interactive Plot - Electric Field Parallel Measurements")
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
        location = selected_date_range['Location']  # Get the location

        
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
            xaxis_title='Time(LT)',
            yaxis=dict(
                title='Electric Field (V/m)',
                range=[min_value, max_value]  # Unified y-axis range
            ),
            yaxis2=dict(
                title='E Field Avg (V/m)',
                overlaying='y',
                side='right',
                range=[min_value, max_value]  # Unified y-axis range
            ),
            legend=dict(x=0.01, y=0.99),
            template='plotly_dark',
            annotations=[  # Add the location as a subtitle
            dict(
                text=f"Location: {location}",
                xref="paper", yref="paper",
                x=0.5, y=1.05,  # Adjust x and y for positioning
                showarrow=False,
                font=dict(size=12)
            )
        ]
        )
        
        # Show the plot in Streamlit
        selected_data = st.plotly_chart(fig, use_container_width=True)

    #sCATTER PLOT
    if selected_date_range:
        start_time = selected_date_range['Start']
        end_time = selected_date_range['End']

        # Filter data for the selected range
        elmagde_filtered = elmagde[(elmagde['timestamp'] >= start_time) & (elmagde['timestamp'] <= end_time)]
        exdf_filtered = exdf[(exdf['timestamp'] >= start_time) & (exdf['timestamp'] <= end_time)]

        # Add a time selection slider for scatter plot
        time_range = st.slider("Select time range for scatter plot:", 
                                min_value=start_time.time(), 
                                max_value=end_time.time(), 
                                value=(start_time.time(), end_time.time()))

        # Create start and end times based on the selected time range
        start_time_scatter = datetime.combine(start_time.date(), time_range[0])
        end_time_scatter = datetime.combine(start_time.date(), time_range[1])

        # Filter both datasets based on the selected time range
        elmagde_scatter = elmagde_filtered[(elmagde_filtered['timestamp'] >= start_time_scatter) & 
                                        (elmagde_filtered['timestamp'] <= end_time_scatter)]
        exdf_scatter = exdf_filtered[(exdf_filtered['timestamp'] >= start_time_scatter) & 
                                    (exdf_filtered['timestamp'] <= end_time_scatter)]

        # Ensure both datasets have the same length
        min_length = min(len(elmagde_scatter), len(exdf_scatter))
        elmagde_scatter = elmagde_scatter.iloc[:min_length]
        exdf_scatter = exdf_scatter.iloc[:min_length]

        # Plot scatter plot if both datasets have data
        if not elmagde_scatter.empty and not exdf_scatter.empty:
            # Create the scatter plot
            plt.figure(figsize=(8, 5))
            plt.scatter(elmagde_scatter['Efield'], exdf_scatter['E_field_Avg'], color='black', alpha=0.6)

            # Fit a linear regression line
            X = elmagde_scatter['Efield'].values.reshape(-1, 1)
            y = exdf_scatter['E_field_Avg'].values
            model = LinearRegression().fit(X, y)
            y_pred = model.predict(X)

            # Plot the linear regression line
            plt.plot(elmagde_scatter['Efield'], y_pred, color='crimson', linewidth=2, label='Fit Line')

            # Calculate and display slope and offset
            slope = model.coef_[0]
            intercept = model.intercept_
            correlation = np.corrcoef(elmagde_scatter['Efield'], exdf_scatter['E_field_Avg'])[0, 1]

            # Add titles and labels
            plt.title('Scatter Plot of Selected Data')
            plt.xlabel('Efield Original EFM (V/m)')
            plt.ylabel('Efield experiment (V/m)')
            plt.grid(True)

            # Display the statistics on the plot
            plt.annotate(f'Correlation: {correlation:.2f}\nSlope: {slope:.2f}\nIntercept: {intercept:.2f}',
                        xy=(0.95, 0.95), xycoords='axes fraction', fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.3', edgecolor='crimson', facecolor='white'))

            # Show the plot in Streamlit
            st.pyplot(plt)

        else:
            st.write("No data available for the selected time range.")