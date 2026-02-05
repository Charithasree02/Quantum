from PIL import Image
import numpy as np

# Create a dummy image
img = Image.new('RGB', (16, 16), color = 'red')
# Quantize to 16 colors
pq = img.quantize(colors=16, method=Image.MEDIANCUT, dither=Image.Dither.NONE)
# Get palette
pal = pq.getpalette()
print(f"Palette length: {len(pal)}")
# Apply FIX: Truncate to 16
pal_arr = np.array(pal).reshape(len(pal)//3, 3)[:16]
print(f"Palette array shape: {pal_arr.shape}")

# Simulate the failing code
try:
    if len(pal_arr) < 16:
        pal_arr = np.vstack([pal_arr, np.zeros((16-len(pal_arr), 3), dtype=np.uint8)])
    
    Y = 0.299*pal_arr[:,0] + 0.587*pal_arr[:,1] + 0.114*pal_arr[:,2]
    C = np.linalg.norm(pal_arr - pal_arr.mean(0), axis=1)
    order = np.lexsort((C, Y))
    print(f"Order shape: {order.shape}")
    
    perm16 = np.empty(16, int)
    print("Attempting: perm16[order] = np.arange(16)")
    perm16[order] = np.arange(16)
    print("Success! Fix verified.")
except Exception as e:
    print(f"FAILED: {e}")
