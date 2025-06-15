import streamlit as st
from streamlit_javascript import st_javascript
from streamlit_folium import st_folium
import folium
from datetime import datetime
import sqlite3
import pandas as pd
import base64
import os
from PIL import Image
import io

# Configuration
DB_NAME = "customer_locations.db"
IMAGE_DIR = "customer_images"
os.makedirs(IMAGE_DIR, exist_ok=True)

def init_db():
    """Initialize database with proper schema"""
    with sqlite3.connect(DB_NAME) as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS locations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                phone TEXT NOT NULL,
                type TEXT NOT NULL,
                address TEXT,
                lat REAL NOT NULL,
                lon REAL NOT NULL,
                notes TEXT,
                image_path TEXT,
                timestamp TEXT NOT NULL
            )
        """)
        conn.commit()
init_db()  # This should already be in your code

# Add this migration code:
with sqlite3.connect(DB_NAME) as conn:
    try:
        # Check if image_path column exists
        conn.execute("SELECT image_path FROM locations LIMIT 1")
    except sqlite3.OperationalError:
        # If not, add it
        conn.execute("ALTER TABLE locations ADD COLUMN image_path TEXT")
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
                (name, phone, type, address, lat, lon, notes, image_path, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                data['name'], data['phone'], data['type'],
                data['address'], data['lat'], data['lon'],
                data['notes'], data.get('image_path'), data['timestamp']
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
st.set_page_config(page_title="Customer Network Builder", layout="wide")
st.title("🏢 Customer Network Builder")

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
with st.expander("📝 Step 2: Customer Information", expanded=True):
    with st.form("customer_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Customer Name*")
            phone = st.text_input("Phone Number*")
            cust_type = st.selectbox("Customer Type*", ["Prospect", "Existing", "VIP", "Repeat"])
            
        with col2:
            address = st.text_area("Address")
            notes = st.text_area("Visit Notes")
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
                    'address': address,
                    'lat': lat,
                    'lon': lon,
                    'notes': notes,
                    'image_path': image_path,
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }):
                    st.success("Customer saved successfully!")
                    st.balloons()
                else:
                    st.error("Failed to save customer data")

# Step 3: Customer Map View
st.subheader("🗺️ Customer Network Map")
data = load_from_db()

if not data.empty:
    # Create map centered on average of all points
    avg_lat = data['lat'].mean()
    avg_lon = data['lon'].mean()
    
    m_all = folium.Map(location=[avg_lat, avg_lon], zoom_start=15, tiles='OpenStreetMap')
    
    # Configure marker colors
    type_colors = {
        "Prospect": "orange",
        "Existing": "green",
        "VIP": "red",
        "Repeat": "blue"
    }
    
    # Add markers
    for _, row in data.iterrows():
        popup_html = f"""
            <div style="width: 150px;">
                {get_image_html(row.get('image_path'))}
                <h4 style="margin: 0;">{row['name']}</h4>
                <p style="margin: 5px 0;">
                    <b>Type:</b> {row['type']}<br>
                    <b>Phone:</b> {row['phone']}<br>
                    <b>Last visit:</b> {row['timestamp']}<br>
                    <i>{row['notes']}</i>
                </p>
            </div>
        """

        folium.Marker(
            [row["lat"], row["lon"]],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=row['name'],
            icon=folium.Icon(color=type_colors.get(row['type'], "gray"))
        ).add_to(m_all)

    # Add heatmap
    from folium.plugins import HeatMap
    heat_data = [[row['lat'], row['lon']] for _, row in data.iterrows()]
    HeatMap(heat_data, radius=15).add_to(m_all)
    
    st_folium(m_all, width=1900, height=800)
else:
    st.info("No customers in your network yet. Start by adding your first customer above.")