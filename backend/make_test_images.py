from PIL import Image, ImageDraw

# official
img = Image.new("RGB", (600, 400), "green")
draw = ImageDraw.Draw(img)
draw.text((180, 180), "SPORTS MATCH", fill="white")
img.save("test_images/official.png")

# edited
img2 = img.copy()
draw2 = ImageDraw.Draw(img2)
draw2.rectangle((20, 20, 220, 80), fill="red")
draw2.text((40, 40), "UNOFFICIAL", fill="white")
img2.save("test_images/edited.png")

print("created")