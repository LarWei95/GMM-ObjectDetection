import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

im = np.array(Image.open('/home/rosvm/data/img/569.jpg'), dtype=np.uint8)

orig_width = 1920
orig_height = 1080

new_width = im.shape[1]
new_height = im.shape[0]
print(im.shape)
height_factor = new_height / orig_height
width_crop = orig_width * height_factor - new_width

print(height_factor, width_crop, orig_width * height_factor)

# Create figure and axes
fig,ax = plt.subplots(1)

# Display the image
ax.imshow(im)

def drawRect(x,y,mx,my,r):
	y *= height_factor
	my *= height_factor

	x = x * height_factor - width_crop/2
	mx = mx * height_factor - width_crop/2
	
	mx -= x
	my -= y

	x -= mx / 2
	y -= my / 2

	return patches.Rectangle((x,y),mx,my,linewidth=1,edgecolor=r,facecolor='none')
# Create a Rectangle patch
#1872.73706055,555.075683594,1882.72374725,575.72407341

ax.add_patch(drawRect(1086.19494629,568.75378418,1152.89877319,622.017593384,'r'))
ax.add_patch(drawRect(806.296813965,597.845581055,995.499313354,681.394058228,'b'))
ax.add_patch(drawRect(924.934204102,574.453918457,1024.89485168,638.389328003,'g'))
ax.add_patch(drawRect(1006.61102295,594.514770508,1143.40892029,696.466468811,'y'))

plt.show()
