import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import math

# ==================== UI THEME ====================
st.set_page_config(page_title="Shape & Contour Analyzer", page_icon="🔷", layout="wide")

st.markdown("""
<style>
body {
    background: linear-gradient(120deg, #0f172a, #020617);
}
.stApp {
    background: linear-gradient(120deg, #0f172a, #020617);
}
h1, h2, h3, h4 {
    color: #f8fafc;
}
.card {
    background: rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 0 30px rgba(0,0,0,0.4);
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("# 🔷 Shape & Contour Analyzer")
st.markdown("Detect, classify, and analyze geometric shapes using computer vision.")
st.markdown("---")

# ==================== SIDEBAR ====================
st.sidebar.header("⚙ Controls")
uploaded = st.sidebar.file_uploader("Upload Image", type=["jpg","png","jpeg"])
min_area = st.sidebar.slider("Minimum Object Area", 100, 5000, 300)

# ==================== SHAPE CLASSIFIER ====================
def classify_shape(cnt):
    area = cv2.contourArea(cnt)
    if area < min_area:
        return None

    perimeter = cv2.arcLength(cnt, True)
    circularity = 4 * math.pi * area / (perimeter * perimeter + 1e-6)

    hull = cv2.convexHull(cnt)
    hull_area = cv2.contourArea(hull)
    solidity = area / (hull_area + 1e-6)

    x, y, w, h = cv2.boundingRect(cnt)
    extent = area / (w * h)

    if len(cnt) >= 5:
        (_, _), (MA, ma), _ = cv2.fitEllipse(cnt)
        ellipse_ratio = min(MA, ma) / max(MA, ma)
    else:
        ellipse_ratio = 0

    if circularity > 0.82 and solidity > 0.9:
        return "Circle"

    if ellipse_ratio > 0.85 and solidity > 0.85:
        return "Ellipse"

    if solidity < 0.8:
        return "Irregular"

    return "Polygon"

# ==================== MAIN ====================
if uploaded:
    img_pil = Image.open(uploaded).convert("RGB")
    img = np.array(img_pil)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='card'><h3>Original Image</h3></div>", unsafe_allow_html=True)
        st.image(img, use_column_width=True)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    output = img.copy()
    results = []
    shape_counts = {}

    for cnt in contours:
        shape = classify_shape(cnt)
        if shape is None:
            continue

        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)

        x,y,w,h = cv2.boundingRect(cnt)
        cv2.drawContours(output, [cnt], -1, (0,255,0), 2)
        cv2.putText(output, shape, (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,0,0),2)

        results.append([shape, round(area,2), round(peri,2)])
        shape_counts[shape] = shape_counts.get(shape, 0) + 1

    with col2:
        st.markdown("<div class='card'><h3>Detected Shapes</h3></div>", unsafe_allow_html=True)
        st.image(output, use_column_width=True)

    st.markdown("---")

    total_objects = sum(shape_counts.values())

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Objects", total_objects)
    c2.metric("Unique Shapes", len(shape_counts))
    c3.metric("Largest Area", max([r[1] for r in results]) if results else 0)

    st.markdown("---")

    st.subheader("📊 Shape Distribution")
    df_summary = pd.DataFrame(shape_counts.items(), columns=["Shape","Count"])
    st.bar_chart(df_summary.set_index("Shape"))

    st.subheader("📋 Measurements")
    df = pd.DataFrame(results, columns=["Shape","Area","Perimeter"])
    st.dataframe(df)

    csv = df.to_csv(index=False).encode()
    st.download_button("⬇ Download CSV", csv, "shape_results.csv", "text/csv")

else:
    st.info("Upload an image from the sidebar to start.")
