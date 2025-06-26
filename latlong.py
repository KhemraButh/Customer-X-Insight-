import streamlit as st
from streamlit_javascript import st_javascript
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import folium
from datetime import datetime
import sqlite3
import pandas as pd
import base64
import os
from PIL import Image
import io
from math import radians, cos, sin, sqrt, atan2
from sklearn.cluster import DBSCAN
import numpy as np


st.set_page_config(page_title="Customer Network Builder", layout="wide")

# Configuration
DB_NAME = "customer_locations.db"
IMAGE_DIR = "/Users/thekhemfee/Downloads/Customer_Network/customer_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

# This should already be in your code

# Add this migration code:
with sqlite3.connect(DB_NAME) as conn:
    try:
        # Check if image_path column exists
        conn.execute("SELECT image_path FROM locations LIMIT 1")
    except sqlite3.OperationalError:
        # If not, add it
        conn.execute("ALTER TABLE locations ADD COLUMN image_path TEXT")
        conn.commit()
    try:
        # Check if RM_Code column exists
        conn.execute("SELECT RM_Code FROM locations LIMIT 1")
    except sqlite3.OperationalError:
        # If not, add it with DEFAULT NULL
        conn.execute("ALTER TABLE locations ADD COLUMN RM_Code TEXT DEFAULT NULL")
        conn.commit()

def save_image(uploaded_file):
    """Save uploaded image to disk and return path"""
    if not uploaded_file:
        return None
        
    # Generate unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = uploaded_file.name.split('.')[-1]
    filename = f"{timestamp}.{ext}"
    filepath = os.path.join(IMAGE_DIR, filename)
    
    # Save the file
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return filepath

def save_to_db(data):
    """Save customer data to database"""
    try:
        with sqlite3.connect(DB_NAME) as conn:
            c = conn.cursor()
            c.execute("""
                INSERT INTO locations 
                (name, phone, type, character, loan_his, bank, amount, term, 
                address, lat, lon, notes, time_visit, image_path, timestamp, RM_Code)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data['name'], data['phone'], data['type'], data['character'],
                data['loan_his'], data['bank'], data['amount'], data['term'],
                data['address'], data['lat'], data['lon'], data['notes'],
                str(data.get('time_visit')), data.get('image_path'), 
                data['timestamp'], data['RM_Code']
            ))
            conn.commit()
            return True
    except sqlite3.Error as e:
        st.error(f"Database error: {str(e)}")
        return False

def load_from_db():
    """Load all customer data from database"""
    with sqlite3.connect(DB_NAME) as conn:
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT * FROM locations")
        rows = c.fetchall()
        return pd.DataFrame([dict(row) for row in rows])

def get_image_html(image_path):
    """Generate HTML for image preview in popup"""
    if not image_path or not os.path.exists(image_path):
        return '<p><i>No image available</i></p>'
    
    try:
        # Create thumbnail
        img = Image.open(image_path)
        img.thumbnail((100, 100))
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG" if image_path.lower().endswith('.jpg') else "PNG")
        encoded = base64.b64encode(buffered.getvalue()).decode('utf-8')
        return f'<img src="data:image/jpeg;base64,{encoded}" style="width: 100px; height: auto; border-radius: 8px; margin-bottom: 5px;" />'
    except Exception as e:
        st.warning(f"Couldn't load image: {str(e)}")
        return '<p><i>Image unavailable</i></p>'


# Streamlit App
col1, col2, col3 = st.columns([1, 6, 1])  # Added empty third column for balance

with col1:
    st.image("/Users/thekhemfee/Downloads/Customer_Network/CusXRealTime/Logo-CMCB_FA-15.png", width=100)

with col2:
    st.markdown("""
        <div style="display: flex; justify-content: center; align-items: center; height: 100%;">
            <h1 style='color: darkgreen;'>CustomerX Insights</h1>
        </div>
    """, unsafe_allow_html=True)

with col3:
    st.empty()

# Step 1: Get GPS
with st.expander("📍 Step 1: Capture Current Location", expanded=False):
    location = st_javascript("""await new Promise((resolve) => {
        navigator.geolocation.getCurrentPosition(
            (pos) => resolve({
                latitude: pos.coords.latitude,
                longitude: pos.coords.longitude
            }),
            (err) => resolve(null)
        );
    })""")

    if location:
        lat = location["latitude"]
        lon = location["longitude"]
        st.success(f"Location captured: Latitude {lat:.6f}, Longitude {lon:.6f}")
        
        # Show current location on small map
        m_current = folium.Map(location=[lat, lon], zoom_start=20, tiles='Esri.WorldImagery')
        folium.Marker(
            [lat, lon], 
            tooltip="Your Location",
            icon=folium.Icon(color="blue", icon="user")
        ).add_to(m_current)
        st_folium(m_current, width=1900, height=800)
    else:
        st.warning("Please enable GPS permissions in your browser to continue")
        st.stop()

# Step 2: Customer Form
with st.expander("📝 Step 2: Customer Information", expanded=False):
    with st.form("customer_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Customer Name*")
            phone = st.text_input("Phone Number*")
            cust_type = st.selectbox("Customer Type*", ["General Retail", "Electronics & Appliances", "Fashion & Clothing", "Restaurant/Café",
            "Street Food Stall", "Beauty Salon", "Repair Services", "Tourism/Hospitality",
            "Agriculture/Farming"])
            #character = st.selectbox("Customer Character*", ["Reliable", "Delayed", "Unknown"])
            character = st.selectbox(
                            "Customer Character*",
                            [
                                "Friendly and Open",
                                "Quiet and Reserved",
                                "Talkative and Engaging",
                                "Skeptical or Distrustful",
                                "Busy or in a Hurry",
                                "Uninterested",
                                "Interested but Cautious",
                                "Very Welcoming",
                                "Aggressive or Pushy",
                                "Hard to Understand"
                            ]
                        )
            loan_his = st.selectbox("Loan History*", ["Used To", "Never Before"])
            bank = st.text_input("Bank Used (Name)*")
            amount = st.number_input("Loan Amount", min_value=0.0, format="%.2f")
            term = st.text_input("Loan Term (e.g. 6 months, 1 year)")
            
            
        with col2:
            #RM_Code = st.selectbox("RM Code*", [])
            RM_Code = st.selectbox("RM Code*", ["RM001", "RM002", "RM003", "RM004", "RM005", "RM006", "RM007", "RM008", "RM009", "RM010"])
            notes = st.text_area("Visit Notes")
            time_visit = st.date_input("Visit Date", value=datetime.now().date())
            address = st.text_input("Address")
            image_file = st.file_uploader("Upload Customer Image", 
                                        type=["png", "jpg", "jpeg"],
                                        accept_multiple_files=False)
            
            if image_file:
                st.image(image_file, caption="Preview", width=150)

        submitted = st.form_submit_button("💾 Save Customer")
        if submitted:
            if not name or not phone:
                st.error("Please fill in all required fields (*)")
            else:
                # Save image and get path
                image_path = save_image(image_file) if image_file else None
                
                if save_to_db({
                    'name': name,
                    'phone': phone,
                    'type': cust_type,
                    'RM_Code': RM_Code,
                    'character': character,
                    'loan_his': loan_his,
                    'bank': bank,
                    'amount': amount,
                    'term': term,
                    'lat': lat,
                    'lon': lon,
                    'notes': notes,
                    'image_path': image_path,
                    'address': address,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }):
                    st.success("Customer saved successfully!")
                    st.balloons()
                else:
                    st.error("Failed to save customer data")


# Step 3: Customer Map View
st.subheader("🗺️ Customer Network Map")
data = load_from_db()
if 'map_center' not in st.session_state:
    st.session_state.map_center = [data["lat"].mean(), data["lon"].mean()]
if 'map_zoom' not in st.session_state:
    st.session_state.map_zoom = 13
if 'click_data' not in st.session_state:
    st.session_state.click_data = None

if not data.empty:
    with st.expander("🔍 Filter Customers"):
        col1, col2 = st.columns(2)
        with col1:
            selected_rm = st.selectbox("Filter by RM Code", ["All"] + sorted(data['RM_Code'].dropna().unique().tolist()))
        with col2:
            selected_type = st.selectbox("Filter by Customer Type", ["All"] + sorted(data['type'].dropna().unique().tolist()))

    # Apply Filters
    filtered_data = data.copy()
    if selected_rm != "All":
        filtered_data = filtered_data[filtered_data['RM_Code'] == selected_rm]
    if selected_type != "All":
        filtered_data = filtered_data[filtered_data['type'] == selected_type]

    # Choose Map Style
    st.radio("🗺️ Map Style", ["Street View", "Satellite View"], key="map_style", horizontal=True)
    tiles = {
        "Street View": "OpenStreetMap",
        "Satellite View": "Esri.WorldImagery"
    }
    selected_tile = tiles[st.session_state.map_style]

    coords = filtered_data[['lat', 'lon']].dropna().values
    kms_per_radian = 6371.0088
    epsilon = 0.5 / kms_per_radian  
    db = DBSCAN(eps=epsilon, min_samples=5, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
    filtered_data['cluster'] = db.labels_
    cluster_summary = filtered_data[filtered_data['cluster'] != -1].groupby('cluster').agg({
        'lat': 'mean',
        'lon': 'mean',
        'name': 'count'
    }).rename(columns={'name': 'count'}).reset_index()

    # Center Map
    center_lat = filtered_data["lat"].mean()
    center_lon = filtered_data["lon"].mean()
    # Initialize zoom level in session state
    if 'map_zoom' not in st.session_state:
        st.session_state.map_zoom = 13 

    # Initial Map (before adding dynamic elements)
    m = folium.Map(location=[center_lat, center_lon], zoom_start=st.session_state.map_zoom, tiles=selected_tile)

    # Customer Type → Marker Color
    type_colors = {
        "General Retail": "blue",
        "Electronics & Appliances": "purple",
        "Fashion & Clothing": "pink",
        "Restaurant/Café": "orange",
        "Street Food Stall": "red",
        "Beauty Salon": "lightgrey",
        "Repair Services": "brown",
        "Tourism/Hospitality": "green",
        "Agriculture/Farming": "darkgreen"
    }

    # Add Markers
    for _, row in filtered_data.iterrows():
        popup_html = f"""
            <div style='width: 200px; font-size: 13px;'>
                {get_image_html(row.get('image_path'))}
                <b>{row['name']}</b><br>
                <b>Type:</b> {row['type']}<br>
                <b>Phone:</b> {row['phone']}<br>
                <b>Character:</b> {row['character']}<br>
                <b>Bank:</b> {row['bank']}<br>
                <b>Loan:</b> ${row['amount']} ({row['term']})<br>
                <b>Visit:</b> {row['time_visit']} on {row['timestamp'][:10]}<br>
                <b>RM:</b> {row.get('RM_Code', 'N/A')}<br>
                <i>{row['notes']}</i>
            </div>
        """
        folium.Marker(
            location=[row["lat"], row["lon"]],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=row["name"],
            icon=folium.Icon(color=type_colors.get(row["type"], "gray"), icon="user")
        ).add_to(m)

    # Heatmap toggle
    if st.toggle("🔥 Show Heatmap"):
        heat_data = filtered_data[["lat", "lon"]].dropna().values.tolist()
        HeatMap(heat_data, radius=30).add_to(m)

    # Zone Suggestion Button
    if st.button("💡 Show Zone Suggestions"):
        for _, row in cluster_summary.iterrows():
            folium.Circle(
                location=[row['lat'], row['lon']],
                radius=500,
                color="blue",
                fill=True,
                fill_opacity=0.4,
                tooltip=f"📍 Zone {row['cluster']} - {row['count']} customers nearby"
            ).add_to(m)

        st.subheader("🔮 Suggested Zones to Visit")
        top_zones = cluster_summary.sort_values(by='count', ascending=False).head(5)
        for i, row in top_zones.iterrows():
            st.markdown(f"""
            - **Zone #{int(row['cluster'])}**:
                - 📍 Location: `{row['lat']:.5f}, {row['lon']:.5f}`
                - 👥 Customers: `{row['count']}`
                - ✅ Suggest: Visit this area
            """)
        # Show 2 customers per cluster
        clustered_data = filtered_data[filtered_data['cluster'] != -1]
        st.markdown("### 🧠 Sample Customers in Each Suggested Zone")
        for cluster_id, group in clustered_data.groupby("cluster"):
            st.markdown(f"#### 📍 Zone {cluster_id} ({len(group)} customers total)")

            # Take first 2 customers (or fewer if less)
            sample_customers = group[['name', 'type', 'phone', 'amount', 'character', 'bank', 'time_visit', 'timestamp']]
            
            # Show as table
            st.dataframe(sample_customers, use_container_width=True)

    show_insight = st.toggle("📊 Show Insights Around Click")

    # If click data exists and insight mode is ON
    if show_insight and st.session_state.get("click_data"):
        folium.Circle(
            location=[st.session_state.click_data['lat'], st.session_state.click_data['lon']],
            radius=1000,
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=0.2,
            tooltip="1 km Radius"
        ).add_to(m)

    # Render the map and capture click
    map_data = st_folium(m, width=1900, height=900, returned_objects=["last_clicked"])

    # If user clicked, update session and rerun
    if map_data.get("last_clicked"):
        st.session_state.click_data = {
            'lat': map_data["last_clicked"]["lat"],
            'lon': map_data["last_clicked"]["lng"]
        }
        if map_data.get("zoom"):
            st.session_state.map_zoom = map_data["zoom"]
        st.rerun()  # immediately rerun with click info

    # Show nearby customers only when insight is on
    if show_insight and st.session_state.get("click_data"):
        clicked_lat = st.session_state.click_data['lat']
        clicked_lon = st.session_state.click_data['lon']

        st.success(f"📍 You clicked: {clicked_lat:.6f}, {clicked_lon:.6f}")

        # Haversine distance function
        from math import radians, sin, cos, sqrt, atan2
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371  # Earth radius in km
            dlat = radians(lat2 - lat1)
            dlon = radians(lon2 - lon1)
            a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
            c = 2 * atan2(sqrt(a), sqrt(1 - a))
            return R * c

        nearby_customers = filtered_data[
            filtered_data.apply(
                lambda row: haversine(clicked_lat, clicked_lon, row['lat'], row['lon']) <= 1,
                axis=1
            )
        ]
        st.write(f"🔍 Found {len(nearby_customers)} customers within 1 km")
        st.dataframe(nearby_customers[['name', 'type', 'phone', 'amount', 'character', 'bank', 'time_visit', 'timestamp']])
    else:
        st.info("🧭 Click on the map to find customers within a 1 km radius.")
