import qrcode
from PIL import Image, ImageDraw

# 1. Generate the base QR with no quiet‑zone
qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=10,
    border=0,                 # <-- remove white border
)
qr.add_data("https://aclanthology.org/2025.findings-acl.1123")
qr.make(fit=True)
qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")

# 2. Load & resize your logo
logo = Image.open("/Users/raymond/Desktop/acl-logo.png")
qr_w, qr_h = qr_img.size
logo_size = int(qr_w * 0.25)
logo = logo.resize((logo_size, logo_size), Image.LANCZOS)

# 3. Compute a hole with extra padding
margin_frac = 0.05
margin     = int(qr_w * margin_frac)
hole_size  = logo_size + 2 * margin
hx0 = (qr_w - hole_size)//2
hy0 = (qr_h - hole_size)//2
hx1, hy1 = hx0 + hole_size, hy0 + hole_size

draw = ImageDraw.Draw(qr_img)
# Use square or circle:
draw.rectangle([(hx0, hy0), (hx1, hy1)], fill="white")
# draw.ellipse([(hx0, hy0), (hx1, hy1)], fill="white")

# 4. Paste your logo centered in that hole
logo_x, logo_y = hx0 + margin, hy0 + margin
qr_img.paste(logo, (logo_x, logo_y), mask=(logo if logo.mode=="RGBA" else None))

# 5. Crop off any remaining white
bbox = qr_img.getbbox()          # bounding box of all non-white pixels
qr_img = qr_img.crop(bbox)

# 6. Save
qr_img.save("acl2025-findings-QR.png")
print("Saved → acl2025-findings-QR.png")
